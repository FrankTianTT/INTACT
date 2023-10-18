from typing import List

from torch.func import jacrev, vmap
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions import Bernoulli
from matplotlib import pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset

from tdfa.models.util import build_parallel_layers
from tdfa.utils.metrics import mean_corr_coef
from tdfa.identifiability.data_ganeration import gen_nonlinear_data
from tdfa.stats.metric import mutual_info_estimation


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
    mask = Bernoulli(logits=logits.clamp(-3, 3)).sample(torch.Size([batch_size]))

    repeated_x = x.unsqueeze(0).expand(y_size, -1, -1)
    masked_x = torch.einsum("boi,obi->obi", mask, repeated_x)

    y_hat = model(masked_x).permute(2, 1, 0)[0]
    return F.mse_loss(y, y_hat, reduction='none'), mask


def get_mask_grad(mask, logits, sampling_loss, x_size, sparse_weight=0.01, theta_weight=0.5):
    num_pos = mask.sum(dim=0)
    num_neg = mask.shape[0] - num_pos
    is_valid = ((num_pos > 0) * (num_neg > 0)).float()
    pos_grads = torch.einsum("sbo,sboi->sboi", sampling_loss, mask).sum(dim=0) / (num_pos + 1e-6)
    neg_grads = torch.einsum("sbo,sboi->sboi", sampling_loss, 1 - mask).sum(dim=0) / (num_neg + 1e-6)

    clip_logits = logits.clamp(-3, 3)
    g = clip_logits.sigmoid() * (1 - clip_logits.sigmoid())
    reg = torch.ones(pos_grads.shape[1:]) * sparse_weight
    reg[:, x_size:] += theta_weight
    grad = is_valid * (pos_grads - neg_grads + reg) * g
    return grad.mean(0)


def train(model, mask_logits, theta_hat, loader, model_optimizer, theta_optimizer, mask_optimizer, steps,
          train_mask_iters=10, train_predictor_iters=50, sampling_times=10):
    predictor_losses = []
    reinforce_losses = []
    for x, y, idx in loader:
        batch_size, x_size = x.shape
        _, y_size = y.shape
        _, theta_size = theta_hat.shape
        x_theta = torch.cat([x, theta_hat[idx]], dim=1)
        if steps % (train_mask_iters + train_predictor_iters) < train_predictor_iters:
            loss, mask = get_model_loss(model, x_theta, y, mask_logits)
            mi_loss = mutual_info_estimation(theta_hat[idx])
            predict_loss = loss.mean() + mi_loss * 1
            # mcc: 0.6234189053823959
            predictor_losses.append(predict_loss.item())

            model_optimizer.zero_grad()
            theta_optimizer.zero_grad()
            predict_loss.backward()
            model_optimizer.step()
            theta_optimizer.step()
        else:  # reinforce
            new_batch_size = batch_size * sampling_times
            repeated_x_theta = x_theta.unsqueeze(0).expand(sampling_times, -1, -1).reshape(new_batch_size, -1)
            repeated_y = y.unsqueeze(0).expand(sampling_times, -1, -1).reshape(new_batch_size, -1)

            loss, mask = get_model_loss(model, repeated_x_theta, repeated_y, mask_logits)
            reinforce_losses.append(loss.mean().item())
            sampling_loss = loss.reshape(sampling_times, batch_size, y_size)
            sampling_mask = mask.reshape(sampling_times, batch_size, y_size, x_size + theta_size)

            grad = get_mask_grad(sampling_mask, mask_logits, sampling_loss, x_size)

            mask_optimizer.zero_grad()
            mask_logits.backward(grad)
            mask_optimizer.step()

        steps += 1
    return steps, predictor_losses, reinforce_losses


def identify_theta(x, y, theta_size):
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    dataset = MyDataset(x, y)
    loader = DataLoader(dataset, batch_size=64)

    task_num, sample_per_task, x_size = x.shape
    *_, y_size = y.shape

    model = build_parallel_layers(x_size + theta_size, 1, [256, 256, 256], extra_dims=[y_size])
    mask_logits = nn.Parameter(torch.randn(y_size, x_size + theta_size))
    theta_hat = torch.nn.Parameter(torch.randn(task_num, theta_size))

    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    theta_optimizer = torch.optim.Adam([theta_hat], lr=0.1)
    mask_optimizer = torch.optim.Adam([mask_logits], lr=0.01)

    steps = 0
    for epoch in range(1000):
        steps, predictor_losses, reinforce_losses = train(model, mask_logits, theta_hat, loader, model_optimizer,
                                                          theta_optimizer, mask_optimizer, steps)

        print(np.mean(predictor_losses), np.mean(reinforce_losses))
        equal = (mask_logits > 0) == torch.from_numpy(graph)
        print(equal[:, :x_size].float().mean(), equal[:, x_size:].float().mean())

        print((mask_logits > 0).int(), (mask_logits > 0)[:, x_size:].sum())

        print("mcc:", mean_corr_coef(theta_hat.detach().numpy(), theta))


        # print(mask_logits[:, x_size:])

        # if epoch % 20 == 0:
        #     sns.displot(theta_hat.detach().numpy()[:, 0], kde=True, bins=20)
        #     plt.show()


if __name__ == '__main__':
    task_num = 1000
    sample_per_task = 100
    x_size = 5
    y_size = 8
    theta_size = 0
    seed = 0

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    x, y, theta, graph = gen_nonlinear_data(task_num, sample_per_task, x_size, y_size, theta_size)

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
