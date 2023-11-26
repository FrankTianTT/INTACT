from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from tdfa.modules.utils import build_parallel_layers
from tdfa.modules.models.context_model import ContextModel
from tdfa.modules.models.causal_mask import CausalMask


class CausalWorldModel(nn.Module):
    def __init__(
            self,
            obs_dim,
            action_dim,
            meta=False,
            max_context_dim=10,
            task_num=100,
            residual=True,
            logits_clip=3.0,
            logits_init_bias=1.0,
            logits_init_scale=0.05,
            stochastic=True,
            hidden_dims=None,
            log_var_bounds=(-10.0, 0.5)
    ):
        """World-model class for environment learning with causal discovery.

        :param obs_dim: number of observation dimensions
        :param action_dim: number of action dimensions
        :param meta: whether to use meta-RL
        :param max_context_dim: number of context dimensions, used for meta-RL, set to 0 if normal RL
        :param task_num: number of tasks, used for meta-RL, set to 0 if normal RL
        :param residual: whether to use residual connection for transition model
        :param logits_clip: clip value for mask logits, default to 3.0 (sigmoid(3.0) = 0.95).
        :param logits_init_bias: bias for mask logits initialization, default to 1.0 (sigmoid(1.0) = 0.73).
        :param logits_init_scale: scale for mask logits initialization, default to 0.05
        :param stochastic: whether to use stochastic transition model (using gaussian nll loss rather than mse loss)
        :param log_var_bounds: bounds for log_var of gaussian nll loss
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.meta = meta
        self.max_context_dim = max_context_dim if meta else 0
        self.residual = residual
        self.stochastic = stochastic
        self.hidden_dims = hidden_dims or [256, 256]
        self.log_var_bounds = log_var_bounds

        # context model
        self.context_model = ContextModel(meta=meta, max_context_dim=max_context_dim, task_num=task_num)

        # causal mask
        self.causal_mask = CausalMask(
            observed_input_dim=self.obs_dim + self.action_dim,
            mask_output_dim=self.output_dim,
            meta=meta,
            context_input_dim=max_context_dim,
            logits_clip=logits_clip,
            logits_init_bias=logits_init_bias,
            logits_init_scale=logits_init_scale,
        )

        # mdp model (transition + reward + termination)
        self.module = build_parallel_layers(
            input_dim=self.all_input_dim,
            output_dim=2,
            hidden_dims=self.hidden_dims,
            extra_dims=[self.output_dim],
            activate_name="ReLU",
        )

        self.original_input_batch_size = None

    def get_parameter(self, target: str):
        if target == "module":
            return self.module.parameters()
        elif target == "mask_logits":
            return self.causal_mask.parameters()
        elif target == "context_hat":
            return self.context_model.parameters()
        else:
            raise NotImplementedError

    @property
    def all_input_dim(self):
        return self.obs_dim + self.action_dim + self.max_context_dim

    @property
    def output_dim(self):
        return self.obs_dim + 2

    def get_log_var(self, log_var):
        min_log_var, max_log_var = self.log_var_bounds
        log_var = max_log_var - F.softplus(max_log_var - log_var)
        log_var = min_log_var + F.softplus(log_var - min_log_var)
        return log_var

    def forward(self, observation, action, idx=None, deterministic_mask=True):
        inputs = torch.cat([observation, action, self.context_model(idx)], dim=-1)
        batch_size, dim = inputs.shape[:-1], inputs.shape[-1]

        masked_inputs, mask = self.causal_mask(inputs.reshape(-1, dim), deterministic=deterministic_mask)
        outputs = self.module(masked_inputs).permute(2, 1, 0)

        next_obs_mean, reward, terminated = outputs[0, :, :-2], outputs[0, :, -2:-1], outputs[0, :, -1:]
        next_obs_log_var = self.get_log_var(outputs[1, :, :-2])

        next_obs_mean = next_obs_mean.reshape(*batch_size, self.obs_dim)
        if self.residual:
            next_obs_mean = observation + next_obs_mean
        next_obs_log_var = next_obs_log_var.reshape(*batch_size, self.obs_dim)
        reward = reward.reshape(*batch_size, 1)
        terminated = terminated.reshape(*batch_size, 1)
        mask = mask.reshape(*batch_size, self.causal_mask.mask_output_dim, self.causal_mask.mask_input_dim)

        return next_obs_mean, next_obs_log_var, reward, terminated, mask


def test_causal_world_model_without_meta():
    obs_dim = 4
    action_dim = 1
    batch_size = 32
    env_num = 5

    world_model = CausalWorldModel(obs_dim=obs_dim, action_dim=action_dim, meta=False)

    for batch_shape in [(), (batch_size,), (env_num, batch_size)]:
        observation = torch.randn(*batch_shape, obs_dim)
        action = torch.randn(*batch_shape, action_dim)

        next_obs_mean, next_obs_log_var, reward, terminated, mask = world_model(observation, action)

        assert next_obs_mean.shape == next_obs_log_var.shape == (*batch_shape, obs_dim)
        assert reward.shape == terminated.shape == (*batch_shape, 1)
        assert mask.shape == (*batch_shape, obs_dim + 2, obs_dim + action_dim)


def test_causal_world_model_with_meta():
    obs_dim = 4
    action_dim = 1
    max_context_dim = 10
    task_num = 100
    batch_size = 32
    env_num = 5

    world_model = CausalWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        meta=True,
        max_context_dim=max_context_dim,
        task_num=task_num
    )

    for batch_shape in [(), (batch_size,), (env_num, batch_size)]:
        observation = torch.randn(*batch_shape, obs_dim)
        action = torch.randn(*batch_shape, action_dim)
        idx = torch.randint(0, task_num, (*batch_shape, 1))

        next_obs_mean, next_obs_log_var, reward, terminated, mask = world_model(observation, action, idx)

        assert next_obs_mean.shape == next_obs_log_var.shape == (*batch_shape, obs_dim)
        assert reward.shape == terminated.shape == (*batch_shape, 1)
        assert mask.shape == (*batch_shape, obs_dim + 2, obs_dim + action_dim + max_context_dim)


if __name__ == '__main__':
    test_causal_world_model_with_meta()
