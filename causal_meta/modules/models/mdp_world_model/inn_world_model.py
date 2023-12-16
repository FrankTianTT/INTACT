from functools import reduce
from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from FrEIA.framework import Node, InputNode, GraphINN, ConditionNode, OutputNode
from FrEIA.modules import PermuteRandom, RNVPCouplingBlock
from FrEIA.modules.coupling_layers import _BaseCouplingBlock

from causal_meta.modules.utils import build_mlp
from causal_meta.modules.models.mdp_world_model.base_world_model import BaseMDPWorldModel
from causal_meta.modules.models.causal_mask import CausalMask


class GINCouplingBlock(_BaseCouplingBlock):

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "ATAN",
                 split_len: Union[float, int] = 0.5):

        super().__init__(dims_in, dims_c, clamp, clamp_activation,
                         split_len=split_len)

        self.subnet1 = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2 * 2)
        self.subnet2 = subnet_constructor(self.split_len2 + self.condition_length, self.split_len1 * 2)

    def _coupling1(self, x1, u2, rev=False):
        a2 = self.subnet2(u2)
        s2, t2 = a2[:, :self.split_len1], a2[:, self.split_len1:]
        s2 = self.clamp * self.f_clamp(s2)

        s2 = s2 - s2.mean(1, keepdim=True)

        if rev:
            y1 = (x1 - t2) * torch.exp(-s2)
            return y1, 0.
        else:
            y1 = torch.exp(s2) * x1 + t2
            return y1, 0.

    def _coupling2(self, x2, u1, rev=False):
        a1 = self.subnet1(u1)
        s1, t1 = a1[:, :self.split_len2], a1[:, self.split_len2:]
        s1 = self.clamp * self.f_clamp(s1)
        s1 = s1 - s1.mean(1, keepdim=True)

        if rev:
            y2 = (x2 - t1) * torch.exp(-s1)
            return y2, 0.
        else:
            y2 = torch.exp(s1) * x2 + t1
            return y2, 0.


class INNWorldModel(BaseMDPWorldModel):
    def __init__(
            self,
            obs_dim,
            action_dim,
            meta=True,
            max_context_dim=4,
            task_num=100,
            residual=True,
            hidden_size=200,
            hidden_layers=2,
    ):
        assert meta, "Meta-RL is required for INNWorldModel"
        max_context_dim = obs_dim + 2

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            meta=True,
            max_context_dim=max_context_dim,
            task_num=task_num,
            residual=residual,
        )

    @property
    def learn_obs_var(self):
        return False

    def build_module(self):
        def subnet_fc(dims_in, dims_out):
            return nn.Sequential(
                nn.Linear(dims_in, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, dims_out)
            )

        nodes = [
            ConditionNode(self.obs_dim + self.action_dim, name='State and Action Input'),
            InputNode(self.obs_dim + 2, name='Context Input')
        ]
        for k in range(self.hidden_layers):
            nodes.append(Node(nodes[-1],
                              GINCouplingBlock,
                              {'subnet_constructor': subnet_fc},
                              conditions=nodes[0],
                              name=f"Coupling_{k}"))
            nodes.append(Node(nodes[-1],
                              PermuteRandom,
                              {'seed': k},
                              name=F'Permute_{k}'))
        nodes.append(OutputNode(nodes[-1], name='Output'))

        return GraphINN(nodes)

    def forward(self, obs, action, idx):
        context = self.context_model(idx)
        batch_size = obs.shape[:-1]

        obs = obs.reshape(-1, self.obs_dim)
        action = action.reshape(-1, self.action_dim)
        context = context.reshape(-1, self.max_context_dim)

        output, log_jac_det = self.module(context, c=torch.cat([obs, action], dim=-1), rev=False)
        next_obs, reward, terminated = output[:, :-2], output[:, -2:-1], output[:, -1:]
        if self.residual:
            next_obs += obs

        next_obs = next_obs.reshape(*batch_size, self.obs_dim)
        reward = reward.reshape(*batch_size, 1)
        terminated = terminated.reshape(*batch_size, 1)

        log_jac_det = log_jac_det.reshape(*batch_size, 1)
        return next_obs, torch.zeros_like(next_obs), reward, terminated, log_jac_det

    def inv_forward(self, obs, action, next_obs, reward, terminated):
        batch_size = obs.shape[:-1]

        obs = obs.reshape(-1, self.obs_dim)
        action = action.reshape(-1, self.action_dim)
        next_obs = next_obs.reshape(-1, self.obs_dim)
        reward = reward.reshape(-1, 1)
        terminated = terminated.reshape(-1, 1)

        if self.residual:
            next_obs = next_obs - obs
        context, log_jac_det = self.module(
            torch.cat([next_obs, reward, terminated], dim=-1),
            c=torch.cat([obs, action], dim=-1),
            rev=True
        )

        context = context.reshape(*batch_size, self.max_context_dim)

        log_jac_det = log_jac_det.reshape(*batch_size, 1)
        return context, log_jac_det


def test_inn_world_model():
    obs_dim = 4
    action_dim = 1
    task_num = 100
    batch_size = 32
    env_num = 5

    world_model = INNWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        task_num=task_num,
    )

    for batch_shape in [(), (batch_size,), (env_num, batch_size)]:
        observation = torch.randn(*batch_shape, obs_dim)
        action = torch.randn(*batch_shape, action_dim)
        idx = torch.randint(0, task_num, (*batch_shape, 1))

        next_obs, _, reward, terminated, log_jac_det = world_model(observation, action, idx)
        context_hat = world_model.context_model(idx)

        assert next_obs.shape == (*batch_shape, obs_dim)
        assert reward.shape == terminated.shape == (*batch_shape, 1)
        assert log_jac_det.shape == (*batch_shape, 1)
        assert context_hat.shape == (*batch_shape, obs_dim + 2)

        context, log_jac_det = world_model.inv_forward(observation, action, next_obs, reward, terminated)

        assert torch.allclose(context, context_hat, atol=1e-6)


if __name__ == '__main__':
    test_inn_world_model()
