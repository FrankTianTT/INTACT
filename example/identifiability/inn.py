import os
import math

import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev, vmap
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from FrEIA.framework import Node, InputNode, GraphINN, ConditionNode, OutputNode
from FrEIA.modules import GINCouplingBlock, PermuteRandom, RNVPCouplingBlock
from tqdm import tqdm

from tdfa.stats.mcc import mean_corr_coef

from env_generation import gen_meta_mdp_data


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(
        nn.Linear(dims_in, 256),
        nn.ReLU(),
        nn.Linear(256, dims_out)
    )


class InvWorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, task_num=100):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.task_num = task_num

        nodes = [
            ConditionNode(obs_dim + action_dim, name='State and Action Input'),
            InputNode(obs_dim, name='Context Input')
        ]
        for k in range(4):
            nodes.append(Node(nodes[-1],
                              RNVPCouplingBlock,
                              {'subnet_constructor': subnet_fc},
                              conditions=nodes[0],
                              name=f"Coupling_{k}"))
            nodes.append(Node(nodes[-1],
                              PermuteRandom,
                              {'seed': k},
                              name=F'Permute_{k}'))
        nodes.append(OutputNode(nodes[-1], name='Output'))

        self.module = GraphINN(nodes)
        self.mu = nn.Parameter(torch.randn(task_num, obs_dim) * 0.05, requires_grad=True)
        self.log_sig = nn.Parameter(torch.zeros(task_num, obs_dim), requires_grad=True)

    def forward(self, x, c=None, rev=False):
        return self.module(x, c=c, rev=rev)

    def init_by_data(self, loader):
        obs, action, next_obs, idx = next(iter(loader))
        device = self.mu.device
        obs, action, next_obs, idx = obs.to(device), action.to(device), next_obs.to(device), idx.to(device)
        idx = idx.squeeze()

        z, log_jac_det = self.module(next_obs, c=torch.cat([obs, action], dim=-1), rev=True)
        self.mu.data = torch.stack([z[idx == i].mean(0) for i in range(self.task_num)])
        self.log_sig.data = torch.stack([z[idx == i].std(0, unbiased=False) for i in range(self.task_num)]).log()


def inn_world_model(
        env_name='CartPoleContinuous-v0',
        task_num=100,
        sample_num=10000,
        batch_size=256,
        lr=1e-3,
        device="cuda",
        empirical_vars=False,
        sparsity_reg=0.0,
):
    obs, action, next_obs, idx, context_dict = gen_meta_mdp_data(
        env_name=env_name,
        task_num=task_num,
        sample_num=sample_num
    )

    model = InvWorldModel(obs.shape[1], action.shape[1], task_num).to(device)

    def f(obs, action, next_obs):
        context, _ = model(next_obs, c=torch.cat([obs, action], dim=-1), rev=True)
        return torch.sum(context, dim=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(obs, action, next_obs, idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # model.init_by_data(dataloader)

    for epoch in range(100):
        losses = []
        mse = []
        recon_losses = []
        jac_losses = []
        for obs, action, next_obs, idx in dataloader:
            obs, action, next_obs, idx = obs.to(device), action.to(device), next_obs.to(device), idx.to(device)
            idx = idx.squeeze()
            model.zero_grad()

            z, log_jac_det = model(next_obs, c=torch.cat([obs, action], dim=-1), rev=True)

            if empirical_vars:
                sig = torch.stack([z[idx == i].std(0, unbiased=False) for i in range(task_num)])
                loss = sig[idx].log().mean(1) - log_jac_det
            else:
                m = model.mu[idx]
                loss = 0.5 * torch.sum((z - m) ** 2, 1) - log_jac_det
                mse.append(torch.mean((z - m) ** 2).item())

                with torch.no_grad():
                    pred_next_state = model(m, c=torch.cat([obs, action], dim=-1), rev=False)[0]
                    recon_loss = nn.functional.mse_loss(pred_next_state, next_obs)
                    recon_losses.append(recon_loss.item())

            if sparsity_reg > 0:
                jac = jacrev(f, argnums=2)(obs, action, next_obs).permute(1, 0, 2)
                loss += jac.abs().mean((1, 2))
                jac_losses.append(jac.abs().mean((1, 2)).mean().item())

            loss = loss.mean()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        print(f"Epoch {epoch}, loss: {np.mean(losses)}")
        print(f"Epoch {epoch}, mse: {np.mean(mse)}")
        print(f"Epoch {epoch}, recon_loss: {np.mean(recon_losses)}")
        print(f"Epoch {epoch}, jac_loss: {np.mean(jac_losses)}")

        context_gt = torch.stack(list(context_dict.values()), dim=-1).detach().cpu().numpy()
        context_hat = model.mu.detach().cpu().numpy()
        mcc, permutation = mean_corr_coef(context_gt, context_hat, return_permutation=True)
        print(f"Epoch {epoch}, mcc: {mcc}")

        os.makedirs("img", exist_ok=True)
        idxes_gt, idxes_hat = permutation

        if len(idxes_gt) == 1:
            plt.scatter(context_gt[:, idxes_gt[0]], context_hat[:, idxes_hat[0]])
        else:
            num_cols = math.isqrt(len(idxes_gt))
            num_rows = math.ceil(len(idxes_gt) / num_cols)
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

            for i, (idx_gt, idx_hat, name) in enumerate(zip(idxes_gt, idxes_hat, context_dict.keys())):
                ax = axs.flatten()[i]
                ax.set(xticks=[], yticks=[], xlabel=name)
                ax.scatter(context_gt[:, idx_gt], context_hat[:, idx_hat])

            for i in range(len(idxes_gt), len(axs.flat)):
                axs.flat[i].set_visible(False)

        plt.savefig(f"img/{epoch}.png")
        plt.close()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    inn_world_model("toy", sparsity_reg=1)
