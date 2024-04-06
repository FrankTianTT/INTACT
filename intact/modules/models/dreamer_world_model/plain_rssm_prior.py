from math import prod

import torch
from torch import nn
from torchrl.modules.distributions import NormalParamWrapper

from intact.modules.models.base_world_model import BaseWorldModel
from intact.modules.utils import build_mlp


class PlainRSSMPrior(BaseWorldModel):
    def __init__(
        self,
        action_dim,
        variable_num=10,
        state_dim_per_variable=3,
        belief_dim_per_variable=20,
        hidden_dim=256,
        disable_belief=False,
        meta=False,
        max_context_dim=10,
        task_num=50,
        residual=True,
        scale_lb=0.1,
    ):
        """
        Class for the Plain RSSM prior.

        Args:
            action_dim (int): the action dimension
            variable_num (int): the number of variables
            state_dim_per_variable (int): the state dimension per variable
            belief_dim_per_variable (int): the belief dimension per variable
            hidden_dim (int): the hidden dimension
            disable_belief (bool): whether to disable belief
            meta (bool): whether to use meta
            max_context_dim (int): the maximum context dimension
            task_num (int): the task number
            residual (bool): whether to use residual
            scale_lb (float): the scale lower bound
        """
        self.variable_num = variable_num
        self.state_dim_per_variable = state_dim_per_variable
        self.belief_dim_per_variable = belief_dim_per_variable
        self.hidden_dim = hidden_dim
        self.disable_belief = disable_belief
        self.scale_lb = scale_lb

        super().__init__(
            obs_dim=self.total_state_dim + self.total_belief_dim,
            action_dim=action_dim,
            meta=meta,
            max_context_dim=max_context_dim,
            task_num=task_num,
            residual=residual,
        )

    @property
    def total_state_dim(self):
        return self.state_dim_per_variable * self.variable_num

    @property
    def total_belief_dim(self):
        return self.belief_dim_per_variable * self.variable_num

    def build_nets(self):
        as2middle = build_mlp(
            input_dim=self.action_dim
            + self.total_state_dim
            + self.max_context_dim,
            output_dim=self.total_belief_dim,
            last_activate_name="ELU",
        )
        middle2s = NormalParamWrapper(
            build_mlp(
                input_dim=self.total_belief_dim,
                output_dim=self.total_state_dim * 2,
                # hidden_dims=[self.hidden_dim] if not self.disable_belief else [self.hidden_dim] * 2,
                hidden_dims=[self.total_belief_dim],
                activate_name="ELU",
            ),
            scale_lb=self.scale_lb,
            scale_mapping="softplus",
        )
        module_dict = nn.ModuleDict(
            dict(as2middle=as2middle, middle2s=middle2s)
        )
        if not self.disable_belief:
            rnn = nn.GRUCell(
                input_size=self.total_belief_dim,
                hidden_size=self.total_belief_dim,
            )
            module_dict["rnn"] = rnn

        return module_dict

    def forward(
        self, state, belief, action, idx=None, deterministic_mask=False
    ):
        projector_inputs = torch.cat(
            [state, action, self.context_model(idx)], dim=-1
        )
        batch_shape = projector_inputs.shape[:-1]

        middle = self.nets["as2middle"](
            projector_inputs.reshape(prod(batch_shape), -1)
        )
        if self.disable_belief:
            next_belief = belief.clone()
            prior_mean, prior_std = self.nets["middle2s"](middle)
        else:
            next_belief = self.nets["rnn"](
                middle, belief.reshape(prod(batch_shape), -1)
            )
            prior_mean, prior_std = self.nets["middle2s"](next_belief)
            next_belief = next_belief.reshape(*batch_shape, -1)

        prior_mean = prior_mean.reshape(*batch_shape, -1)
        prior_std = prior_std.reshape(*batch_shape, -1)
        if self.residual:
            prior_mean = prior_mean + state

        next_state = prior_mean + torch.randn_like(prior_std) * prior_std

        return prior_mean, prior_std, next_state, next_belief
