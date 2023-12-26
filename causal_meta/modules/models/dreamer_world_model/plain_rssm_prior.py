from math import prod

import torch
from torch import nn
from torchrl.modules.distributions import NormalParamWrapper

from causal_meta.modules.utils import build_mlp
from causal_meta.modules.models.base_world_model import BaseWorldModel


class PlainRSSMPrior(BaseWorldModel):
    def __init__(
            self,
            action_dim,
            variable_num=10,
            state_dim_per_variable=3,
            hidden_dim_per_variable=20,
            rnn_input_dim_per_variable=20,
            meta=False,
            max_context_dim=10,
            task_num=50,
            residual=True,
            scale_lb=0.1,
    ):
        self.variable_num = variable_num
        self.state_dim_per_variable = state_dim_per_variable
        self.hidden_dim_per_variable = hidden_dim_per_variable
        self.rnn_input_dim_per_variable = rnn_input_dim_per_variable
        self.scale_lb = scale_lb

        super().__init__(
            obs_dim=self.total_state_dim + self.total_hidden_dim,
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
    def total_hidden_dim(self):
        return self.hidden_dim_per_variable * self.variable_num

    def build_nets(self):
        action_state_projector = build_mlp(
            input_dim=self.action_dim + self.total_state_dim + self.max_context_dim,
            output_dim=self.rnn_input_dim_per_variable * self.variable_num,
            last_activate_name="ELU",
        )
        rnn = nn.GRUCell(
            input_size=self.rnn_input_dim_per_variable * self.variable_num,
            hidden_size=self.total_hidden_dim,
        )
        rnn_to_prior_projector = NormalParamWrapper(
            build_mlp(
                input_dim=self.total_hidden_dim,
                output_dim=self.total_state_dim * 2,
                hidden_dims=[self.total_hidden_dim],
                activate_name="ELU",
            ),
            scale_lb=self.scale_lb,
            scale_mapping="softplus",
        )
        return nn.ModuleDict(dict(
            as2rnn=action_state_projector,
            rnn=rnn,
            rnn2b=rnn_to_prior_projector,
        ))

    def forward(self, state, belief, action, idx=None, deterministic_mask=False):
        projector_inputs = torch.cat([state, action, self.context_model(idx)], dim=-1)
        batch_shape = projector_inputs.shape[:-1]

        rnn_inputs = self.nets["as2rnn"](projector_inputs.reshape(prod(batch_shape), -1))
        next_belief = self.nets["rnn"](rnn_inputs, belief.reshape(prod(batch_shape), -1))
        prior_mean, prior_std = self.nets["rnn2b"](next_belief)

        prior_mean = prior_mean.reshape(*batch_shape, -1)
        prior_std = prior_std.reshape(*batch_shape, -1)
        next_belief = next_belief.reshape(*batch_shape, -1)
        if self.residual:
            prior_mean = prior_mean + state

        next_state = prior_mean + torch.randn_like(prior_std) * prior_std

        return prior_mean, prior_std, next_state, next_belief


def test_plain_rssm_prior():
    action_dim = 1
    variable_num = 10
    state_dim_per_variable = 3
    hidden_dim_per_variable = 20
    max_context_dim = 10
    task_num = 50
    env_num = 10
    batch_size = 32

    prior = PlainRSSMPrior(
        action_dim=action_dim,
        variable_num=variable_num,
        state_dim_per_variable=state_dim_per_variable,
        hidden_dim_per_variable=hidden_dim_per_variable,
        max_context_dim=max_context_dim,
        task_num=task_num,
        meta=True
    )

    for batch_shape in [(), (batch_size,), (env_num, batch_size)]:
        state = torch.randn(*batch_shape, variable_num * state_dim_per_variable)
        belief = torch.randn(*batch_shape, variable_num * hidden_dim_per_variable)
        action = torch.randn(*batch_shape, action_dim)
        idx = torch.randint(0, task_num, (*batch_shape, 1))

        prior_mean, prior_std, next_state, next_belief = prior(state, belief, action, idx)

        assert prior_mean.shape == prior_std.shape == next_state.shape == state.shape
        assert next_belief.shape == belief.shape


if __name__ == '__main__':
    test_plain_rssm_prior()
