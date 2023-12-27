from itertools import chain
from math import prod

import torch
from torch import nn
from torchrl.modules.distributions import NormalParamWrapper

from causal_meta.modules.utils import build_mlp
from causal_meta.modules.models.layers import ParallelGRUCell
from causal_meta.modules.models.causal_mask import CausalMask
from causal_meta.modules.models.dreamer_world_model import PlainRSSMPrior


class CausalRSSMPrior(PlainRSSMPrior):
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
            logits_clip=3.0,
            scale_lb=0.1,
    ):
        self.logits_clip = logits_clip

        super().__init__(
            action_dim=action_dim,
            variable_num=variable_num,
            state_dim_per_variable=state_dim_per_variable,
            hidden_dim_per_variable=hidden_dim_per_variable,
            rnn_input_dim_per_variable=rnn_input_dim_per_variable,
            meta=meta,
            max_context_dim=max_context_dim,
            task_num=task_num,
            residual=residual,
            scale_lb=scale_lb,
        )

        self.causal_mask = CausalMask(
            observed_input_dim=self.variable_num + self.action_dim,
            context_input_dim=self.max_context_dim,
            mask_output_dim=self.variable_num,
            logits_clip=self.logits_clip,
            meta=self.meta,
            observed_logits_init_bias=3.
        )
        mask_dim_list = []
        for i in range(self.variable_num):
            mask_dim_list.extend([i] * self.state_dim_per_variable)
        mask_dim_list.extend(torch.arange(self.variable_num, self.causal_mask.mask_input_dim).tolist())
        self.mask_dim_map = torch.Tensor(mask_dim_list).long()

    def build_nets(self):
        action_state_projector = build_mlp(
            input_dim=self.action_dim + self.total_state_dim + self.max_context_dim,
            output_dim=self.rnn_input_dim_per_variable,
            extra_dims=[self.variable_num],
            last_activate_name="ELU",
        )
        rnn = ParallelGRUCell(
            input_size=self.rnn_input_dim_per_variable,
            hidden_size=self.hidden_dim_per_variable,
            extra_dims=[self.variable_num],
        )
        rnn_to_prior_projector = NormalParamWrapper(
            build_mlp(
                input_dim=self.hidden_dim_per_variable,
                output_dim=self.state_dim_per_variable * 2,
                extra_dims=[self.variable_num],
                hidden_dims=[self.hidden_dim_per_variable],
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

    @property
    def params_dict(self):
        return dict(
            nets=self.nets.parameters(),
            context=self.context_model.parameters(),
            observed_logits=self.causal_mask.get_parameter("observed_logits"),
            context_logits=self.causal_mask.get_parameter("context_logits"),
        )

    def forward(self, state, belief, action, idx=None, deterministic_mask=False):
        projector_inputs = torch.cat([state, action, self.context_model(idx)], dim=-1)
        batch_shape, dim = projector_inputs.shape[:-1], projector_inputs.shape[-1]

        masked_projector_inputs, mask = self.causal_mask(
            inputs=projector_inputs.reshape(prod(batch_shape), -1),  # (prod(batch_size), input_dim)
            dim_map=self.mask_dim_map,  # input_dim
            deterministic=deterministic_mask
        )  # (variable_num, prod(batch_size), input_dim)

        rnn_inputs = self.nets["as2rnn"](masked_projector_inputs)
        reshaped_belief = belief.reshape(prod(batch_shape), self.variable_num, self.hidden_dim_per_variable)
        reshaped_belief = reshaped_belief.permute(1, 0, 2)
        next_belief = self.nets["rnn"](rnn_inputs, reshaped_belief)
        prior_mean, prior_std = self.nets["rnn2b"](next_belief)

        prior_mean = prior_mean.permute(1, 0, 2).reshape(*batch_shape, -1)
        prior_std = prior_std.permute(1, 0, 2).reshape(*batch_shape, -1)
        next_belief = next_belief.permute(1, 0, 2).reshape(*batch_shape, -1)

        if self.residual:
            prior_mean = prior_mean + state
        next_state = prior_mean + torch.randn_like(prior_std) * prior_std

        return prior_mean, prior_std, next_state, next_belief, mask


def test_causal_rssm_prior():
    action_dim = 1
    variable_num = 10
    state_dim_per_variable = 3
    hidden_dim_per_variable = 20
    max_context_dim = 10
    task_num = 50
    env_num = 10
    batch_size = 32

    prior = CausalRSSMPrior(
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

        prior_mean, prior_std, next_state, next_belief, mask = prior(state, belief, action, idx)

        assert prior_mean.shape == prior_std.shape == next_state.shape == state.shape
        assert next_belief.shape == belief.shape


if __name__ == '__main__':
    test_causal_rssm_prior()
