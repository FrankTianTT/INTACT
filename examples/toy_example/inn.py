import os
import math
from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev, vmap
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from FrEIA.framework import Node, InputNode, GraphINN, ConditionNode, OutputNode
from FrEIA.modules import PermuteRandom, RNVPCouplingBlock
from FrEIA.modules.coupling_layers import _BaseCouplingBlock
from tqdm import tqdm

from causal_meta.stats.mcc import mean_corr_coef

from env_generation import gen_meta_mdp_data


class GINCouplingBlock(_BaseCouplingBlock):
    def __init__(
        self,
        dims_in,
        dims_c=[],
        subnet_constructor: Callable = None,
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "ATAN",
        split_len: Union[float, int] = 0.5,
    ):

        super().__init__(dims_in, dims_c, clamp, clamp_activation, split_len=split_len)

        self.subnet1 = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2 * 2)
        self.subnet2 = subnet_constructor(self.split_len2 + self.condition_length, self.split_len1 * 2)

    def _coupling1(self, x1, u2, rev=False):
        a2 = self.subnet2(u2)
        s2, t2 = a2[:, : self.split_len1], a2[:, self.split_len1 :]
        s2 = self.clamp * self.f_clamp(s2)

        s2 = s2 - s2.mean(1, keepdim=True)

        if rev:
            y1 = (x1 - t2) * torch.exp(-s2)
            return y1, 0.0
        else:
            y1 = torch.exp(s2) * x1 + t2
            return y1, 0.0

    def _coupling2(self, x2, u1, rev=False):
        a1 = self.subnet1(u1)
        s1, t1 = a1[:, : self.split_len2], a1[:, self.split_len2 :]
        s1 = self.clamp * self.f_clamp(s1)
        s1 = s1 - s1.mean(1, keepdim=True)

        if rev:
            y2 = (x2 - t1) * torch.exp(-s1)
            return y2, 0.0
        else:
            y2 = torch.exp(s1) * x2 + t1
            return y2, 0.0


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 256), nn.ReLU(), nn.Linear(256, dims_out))


class InvWorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, task_num=100, residual=True):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.task_num = task_num
        self.residual = residual

        nodes = [ConditionNode(obs_dim + action_dim, name="State and Action Input"), InputNode(obs_dim, name="Context Input")]
        for k in range(4):
            nodes.append(
                Node(nodes[-1], GINCouplingBlock, {"subnet_constructor": subnet_fc}, conditions=nodes[0], name=f"Coupling_{k}")
            )
            nodes.append(Node(nodes[-1], PermuteRandom, {"seed": k}, name=f"Permute_{k}"))
        nodes.append(OutputNode(nodes[-1], name="Output"))

        self.module = GraphINN(nodes)
        self.mu = nn.Parameter(torch.randn(task_num, obs_dim) * 0.05, requires_grad=True)
        self.log_sig = nn.Parameter(torch.zeros(task_num, obs_dim), requires_grad=True)

    def forward(self, obs, action, context, return_det=False):
        next_obs, log_jac_det = self.module(context, c=torch.cat([obs, action], dim=-1), rev=False)
        if self.residual:
            next_obs += obs
        if return_det:
            return next_obs, log_jac_det
        else:
            return next_obs

    def inv_forward(self, obs, action, next_obs, return_det=False):
        if self.residual:
            next_obs = next_obs - obs
        context, log_jac_det = self.module(next_obs, c=torch.cat([obs, action], dim=-1), rev=True)
        if return_det:
            return context, log_jac_det
        else:
            return context

    def init_by_data(self, loader):
        obs, action, next_obs, idx = next(iter(loader))
        device = self.mu.device
        obs, action, next_obs, idx = obs.to(device), action.to(device), next_obs.to(device), idx.to(device)
        idx = idx.squeeze()

        z, log_jac_det = self.module(next_obs, c=torch.cat([obs, action], dim=-1), rev=True)
        self.mu.data = torch.stack([z[idx == i].mean(0) for i in range(self.task_num)])
        self.log_sig.data = torch.stack([z[idx == i].std(0, unbiased=False) for i in range(self.task_num)]).log()


def inn_world_model(
    env_name="CartPoleContinuous-v0",
    task_num=100,
    sample_num=10000,
    batch_size=256,
    lr=1e-3,
    device="cuda",
    empirical_vars=False,
    sparsity_reg=0.0,
):
    if not torch.cuda.is_available():
        device = "cpu"

    obs, action, next_obs, idx, context_dict = gen_meta_mdp_data(env_name=env_name, task_num=task_num, sample_num=sample_num)

    model = InvWorldModel(obs.shape[1], action.shape[1], task_num).to(device)

    def f(obs, action, next_obs):
        context, _ = model.inv_forward(obs, action, next_obs, return_det=True)
        return torch.sum(context, dim=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(obs, action, next_obs, idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # model.init_by_data(dataloader)

    for epoch in range(1000):
        losses = []
        mse = []
        recon_losses = []
        jac_losses = []
        for obs, action, next_obs, idx in dataloader:
            obs, action, next_obs, idx = obs.to(device), action.to(device), next_obs.to(device), idx.to(device)
            idx = idx.squeeze()
            model.zero_grad()

            z, log_jac_det = model.inv_forward(obs, action, next_obs, return_det=True)

            if empirical_vars:
                sig = torch.stack([z[idx == i].std(0, unbiased=False) for i in range(task_num)])
                loss = sig[idx].log().mean(1) - log_jac_det
            else:
                m = model.mu[idx]
                loss = 0.5 * torch.sum((z - m) ** 2, 1) - log_jac_det
                mse.append(torch.mean((z - m) ** 2).item())

                with torch.no_grad():
                    pred_next_state = model(obs, action, m)
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
        mcc, permutation = mean_corr_coef(context_gt, context_hat, return_permutation=True, method="spearman")
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


def mlp(
    env_name="CartPoleContinuous-v0",
    task_num=100,
    sample_num=10000,
    batch_size=256,
    context_dim=1,
    lr=1e-3,
    device="cuda",
):
    if not torch.cuda.is_available():
        device = "cpu"

    obs, action, next_obs, idx, context_dict = gen_meta_mdp_data(env_name=env_name, task_num=task_num, sample_num=sample_num)
    context = nn.Parameter(torch.zeros(task_num, context_dim), requires_grad=True)

    model = nn.Sequential(
        nn.Linear(obs.shape[1] + action.shape[1] + context_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, obs.shape[1]),
    )

    optimizer = torch.optim.Adam(list(model.parameters()) + [context], lr=lr)

    dataset = TensorDataset(obs, action, next_obs, idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(1000):
        losses = []

        for obs, action, next_obs, idx in dataloader:
            obs, action, next_obs, idx = obs.to(device), action.to(device), next_obs.to(device), idx.to(device)
            model.zero_grad()

            pred_next_obs = model(torch.cat([obs, action, context[idx[:, 0]]], dim=-1))

            loss = nn.functional.mse_loss(next_obs, pred_next_obs + obs)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch}, loss: {np.mean(losses)}")

        if epoch % 50 == 0:
            context_gt = torch.stack(list(context_dict.values()), dim=-1).detach().cpu().numpy()
            context_hat = context.detach().numpy()
            plt.scatter(context_gt.reshape(-1), context_hat.reshape(-1))
            plt.show()
            mcc = mean_corr_coef(context_gt, context_hat)
            print(mcc)


if __name__ == "__main__":
    mlp()
