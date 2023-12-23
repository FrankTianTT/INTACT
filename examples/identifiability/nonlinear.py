import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader, Dataset

from causal_meta.modules.utils import build_mlp
from causal_meta.stats.mcc import mean_corr_coef
from examples.identifiability.data_ganeration import gen_nonlinear_data
from causal_meta.stats.metric import mutual_info_estimation
from causal_meta.utils.functional import total_mask_grad


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.task_num, self.sample_per_task, self.x_size = x.shape
        *_, self.y_size = y.shape

        self.x = x.reshape(-1, self.x_size)
        self.y = y.reshape(-1, self.y_size)

        self.permutation = torch.randperm(self.task_num * self.sample_per_task)

    def __len__(self):
        return self.task_num * self.sample_per_task

    def __getitem__(self, item):
        idx = self.permutation[item]
        task_idx = idx // self.sample_per_task
        return self.x[idx], self.y[idx], task_idx


def get_model_loss(model, x, y, logits):
    batch_size, x_size = x.shape
    *_, y_size = y.shape
    mask = Bernoulli(logits=logits).sample(torch.Size([batch_size]))

    repeated_x = x.unsqueeze(0).expand(y_size, -1, -1)
    masked_x = torch.einsum("boi,obi->obi", mask, repeated_x)

    y_hat = model(masked_x).permute(2, 1, 0)[0]
    return F.mse_loss(y, y_hat, reduction='none'), mask


def train(model, mask_logits, theta_hat, loader, model_optimizer, theta_optimizer, mask_optimizer, steps,
          train_mask_iters=10, train_predictor_iters=50, sampling_times=50,
          lambda_mutual_info=0):
    predictor_losses = []
    mi_losses = []
    reinforce_losses = []
    for x, y, idx in loader:
        batch_size, x_size = x.shape
        _, y_size = y.shape
        _, theta_size = theta_hat.shape
        x_theta = torch.cat([x, theta_hat[idx]], dim=1)
        if steps % (train_mask_iters + train_predictor_iters) < train_predictor_iters:
            model_optimizer.zero_grad()
            theta_optimizer.zero_grad()

            loss, mask = get_model_loss(model, x_theta, y, mask_logits)
            if lambda_mutual_info > 0:
                mi_loss = mutual_info_estimation(theta_hat[idx][:, (mask_logits > 0)[:, x_size:].any(dim=0)])
            else:
                mi_loss = 0
            mi_losses.append(mi_loss.item())
            predict_loss = loss.mean() + mi_loss * lambda_mutual_info
            predictor_losses.append(predict_loss.item())

            predict_loss.backward()
            model_optimizer.step()
            theta_optimizer.step()
        else:  # reinforce
            mask_optimizer.zero_grad()

            new_batch_size = batch_size * sampling_times
            repeated_x_theta = x_theta.unsqueeze(0).expand(sampling_times, -1, -1).reshape(new_batch_size, -1)
            repeated_y = y.unsqueeze(0).expand(sampling_times, -1, -1).reshape(new_batch_size, -1)

            loss, mask = get_model_loss(model, repeated_x_theta, repeated_y, mask_logits)
            reinforce_losses.append(loss.mean().item())
            sampling_loss = loss.reshape(sampling_times, batch_size, y_size)
            sampling_mask = mask.reshape(sampling_times, batch_size, y_size, x_size + theta_size)

            grad = total_mask_grad(
                logits=mask_logits,
                sampling_mask=sampling_mask,
                sampling_loss=sampling_loss,
                observed_input_dim=x_size,
                sparse_weight=0.05,
                context_sparse_weight=0.05,
                context_max_weight=1
            )

            mask_logits.backward(grad)
            mask_optimizer.step()

        steps += 1
    return steps, predictor_losses, reinforce_losses, mi_losses


def identify_theta(x, y, theta_size):
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    dataset = MyDataset(x, y)
    loader = DataLoader(dataset, batch_size=64)

    task_num, sample_per_task, x_size = x.shape
    *_, y_size = y.shape

    model = build_mlp(x_size + theta_size, 1, [256, 256], extra_dims=[y_size])
    mask_logits = nn.Parameter(torch.randn(y_size, x_size + theta_size))
    theta_hat = torch.nn.Parameter(torch.randn(task_num, theta_size))

    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    theta_optimizer = torch.optim.Adam([theta_hat], lr=0.1)
    mask_optimizer = torch.optim.Adam([mask_logits], lr=0.05)

    steps = 0
    for epoch in range(1000):
        steps, predictor_losses, reinforce_losses, mi_losses = train(model, mask_logits, theta_hat, loader,
                                                                     model_optimizer,
                                                                     theta_optimizer, mask_optimizer, steps)

        print((mask_logits > 0).int())

        print(np.mean(predictor_losses), np.mean(mi_losses))

        valid_context_idx = torch.where((mask_logits > 0)[:, x_size:].any(dim=0))[0]
        valid_theta_hat = theta_hat[:, valid_context_idx].detach().numpy()
        mcc, (_, permutation) = mean_corr_coef(valid_theta_hat, theta, return_permutation=True)

        print("valid context num:", len(valid_context_idx), "mcc:", mcc)
        print("permutation:", valid_context_idx[permutation])


if __name__ == '__main__':
    task_num = 300
    sample_per_task = 30
    x_size = 5
    y_size = 8
    theta_size = 10
    real_theta_size = 2
    seed = 1
    print(seed)

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    x, y, theta, graph = gen_nonlinear_data(task_num, sample_per_task, x_size, y_size, real_theta_size)

    print(graph)
    # [[1 0 0 1 1 1 0]
    #  [0 1 0 0 0 1 1]
    #  [0 0 1 1 1 0 1]
    #  [1 1 1 0 1 0 0]
    #  [1 0 0 0 0 1 1]
    #  [1 1 1 0 0 1 1]
    #  [0 1 0 1 0 1 1]
    #  [0 1 1 1 0 1 0]]

    theta_hat = identify_theta(x, y, theta_size)
    # print(mean_corr_coef(theta_hat, theta))

# [[1 1 1 1 0 0 1]
#  [1 0 0 0 0 0 1]
#  [0 1 1 0 0 0 0]
#  [0 1 1 1 0 1 0]
#  [0 1 0 1 0 0 0]
#  [1 0 0 0 1 1 0]
#  [1 0 1 0 1 0 0]
#  [0 0 1 1 1 0 1]]
