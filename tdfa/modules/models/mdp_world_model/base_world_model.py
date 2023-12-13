from functools import reduce
from copy import deepcopy
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from tdfa.modules.utils import build_mlp
from tdfa.modules.models.context_model import ContextModel
from tdfa.modules.models.causal_mask import CausalMask


class BaseMDPWorldModel(nn.Module):
    def __init__(
            self,
            obs_dim,
            action_dim,
            meta=False,
            max_context_dim=10,
            task_num=100,
            residual=True,
    ):
        """World-model class for environment learning with causal discovery.

        :param obs_dim: number of observation dimensions
        :param action_dim: number of action dimensions
        :param meta: whether to use meta-RL
        :param max_context_dim: number of context dimensions, used for meta-RL, set to 0 if normal RL
        :param task_num: number of tasks, used for meta-RL, set to 0 if normal RL
        :param residual: whether to use residual connection for transition model
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.meta = meta
        self.max_context_dim = max_context_dim if meta else 0
        self.task_num = task_num
        self.residual = residual

        # context model
        self.context_model = ContextModel(meta=meta, max_context_dim=max_context_dim, task_num=task_num)

        # mdp model (transition + reward + termination)
        self.module = self.build_module()

    @property
    @abstractmethod
    def learn_obs_var(self):
        raise NotImplementedError

    @abstractmethod
    def build_module(self):
        raise NotImplementedError

    @property
    def params_dict(self):
        return dict(
            module=self.module.parameters(),
            context=self.context_model.parameters()
        )

    def get_parameter(self, target: str):
        if target in self.params_dict:
            return self.params_dict[target]
        else:
            raise NotImplementedError

    @property
    def all_input_dim(self):
        return self.obs_dim + self.action_dim + self.max_context_dim

    @property
    def output_dim(self):
        return self.obs_dim + 2

    def freeze_module(self):
        for name in self.params_dict.keys():
            if name == "context":
                continue
            for param in self.params_dict[name]:
                param.requires_grad = False

    def reset_context(self, task_num=None):
        self.context_model.reset(task_num)


class PlainMDPWorldModel(BaseMDPWorldModel):
    def __init__(
            self,
            obs_dim,
            action_dim,
            meta=False,
            max_context_dim=10,
            task_num=100,
            residual=True,
            learn_obs_var=True,
            hidden_dims=None,
            log_var_bounds=(-10.0, -2.0)
    ):
        """World-model class for environment learning with causal discovery.

        :param obs_dim: number of observation dimensions
        :param action_dim: number of action dimensions
        :param meta: whether to use meta-RL
        :param max_context_dim: number of context dimensions, used for meta-RL, set to 0 if normal RL
        :param task_num: number of tasks, used for meta-RL, set to 0 if normal RL
        :param residual: whether to use residual connection for transition model
        :param hidden_dims: hidden dimensions for transition model
        :param log_var_bounds: bounds for log_var of gaussian nll loss
        """

        self.hidden_dims = hidden_dims or [256, 256]
        self.log_var_bounds = log_var_bounds
        self._learn_obs_var = learn_obs_var

        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            meta=meta,
            max_context_dim=max_context_dim,
            task_num=task_num,
            residual=residual,
        )

    @property
    def learn_obs_var(self):
        return self._learn_obs_var

    def build_module(self):
        return build_mlp(
            input_dim=self.all_input_dim,
            output_dim=self.output_dim * 2,
            hidden_dims=self.hidden_dims,
            extra_dims=None,
            activate_name="ReLU",
        )

    def get_log_var(self, log_var):
        min_log_var, max_log_var = self.log_var_bounds
        log_var = max_log_var - F.softplus(max_log_var - log_var)
        log_var = min_log_var + F.softplus(log_var - min_log_var)
        return log_var

    def get_outputs(self, mean, log_var, observation, batch_size):
        next_obs_mean, reward, terminated = mean[:, :-2], mean[:, -2:-1], mean[:, -1:]
        if self.learn_obs_var:
            next_obs_log_var = self.get_log_var(log_var[:, :-2])
        else:
            next_obs_log_var = torch.zeros_like(log_var[:, :-2])

        next_obs_mean = next_obs_mean.reshape(*batch_size, self.obs_dim)
        if self.residual:
            next_obs_mean = observation + next_obs_mean
        next_obs_log_var = next_obs_log_var.reshape(*batch_size, self.obs_dim)
        reward = reward.reshape(*batch_size, 1)
        terminated = terminated.reshape(*batch_size, 1)

        return next_obs_mean, next_obs_log_var, reward, terminated

    def forward(self, observation, action, idx=None):
        context = self.context_model(idx)

        inputs = torch.cat([observation, action, context], dim=-1)
        batch_size, dim = inputs.shape[:-1], inputs.shape[-1]

        mean, log_var = self.module(inputs.reshape(-1, dim)).chunk(2, dim=-1)
        return self.get_outputs(mean, log_var, observation, batch_size)


def test_plain_world_model():
    obs_dim = 4
    action_dim = 1
    batch_size = 32
    env_num = 5

    world_model = PlainMDPWorldModel(obs_dim=obs_dim, action_dim=action_dim, meta=False)

    for batch_shape in [(), (batch_size,), (env_num, batch_size)]:
        observation = torch.randn(*batch_shape, obs_dim)
        action = torch.randn(*batch_shape, action_dim)

        next_obs_mean, next_obs_log_var, reward, terminated = world_model(observation, action)

        assert next_obs_mean.shape == next_obs_log_var.shape == (*batch_shape, obs_dim)
        assert reward.shape == terminated.shape == (*batch_shape, 1)


def test_reset_context():
    from torch.optim import Adam

    obs_dim = 4
    action_dim = 1
    task_num = 100
    new_task_num = 10
    max_context_dim = 10
    batch_size = 32

    world_model = PlainMDPWorldModel(obs_dim=obs_dim, action_dim=action_dim, meta=True,
                                     task_num=task_num, max_context_dim=max_context_dim)
    cloned_world_model = deepcopy(world_model)
    optimizer1 = Adam(cloned_world_model.get_parameter("context"), lr=0.001)
    cloned_world_model.reset_context(new_task_num)
    optimizer2 = Adam(cloned_world_model.get_parameter("context"), lr=0.001)

    observation = torch.randn(batch_size, obs_dim)
    action = torch.randn(batch_size, action_dim)
    idx = torch.randint(0, new_task_num, (batch_size, 1))
    next_observation = torch.randn(batch_size, obs_dim)

    print(cloned_world_model.context_model.context_hat)
    cloned_world_model.zero_grad()
    next_obs_mean, next_obs_log_var, reward, terminated = cloned_world_model(observation, action, idx)
    loss = nn.functional.mse_loss(next_observation, next_obs_mean)
    loss.backward()
    optimizer1.step()
    print(cloned_world_model.context_model.context_hat)

    # for name, p in world_model.named_parameters():
    #     if name == "context_model.context_hat":
    #         assert p.shape == (task_num, max_context_dim)
    #         assert cloned_world_model.state_dict()[name].shape == (new_task_num, max_context_dim)
    #     else:
    #         assert (p == cloned_world_model.state_dict()[name]).all()


if __name__ == '__main__':
    test_reset_context()
