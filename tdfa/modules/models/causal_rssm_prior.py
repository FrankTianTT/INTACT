from itertools import chain

import torch
from torch import nn
from torchrl.modules.distributions import NormalParamWrapper

from tdfa.modules.utils import build_parallel_layers
from tdfa.modules.models.layers import ParallelGRUCell
from tdfa.modules.models.context_model import ContextModel
from tdfa.modules.models.causal_mask import CausalMask


class CausalRSSMPrior(nn.Module):
    def __init__(
            self,
            action_dim,
            variable_num=10,
            state_dim_per_variable=3,
            hidden_dim_per_variable=20,
            rnn_input_dim_per_variable=20,
            max_context_dim=0,
            task_num=0,
            residual=True,
            logits_clip=3.0,
            scale_lb=0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.variable_num = variable_num
        self.state_dim_per_variable = state_dim_per_variable
        self.hidden_dim_per_variable = hidden_dim_per_variable
        self.rnn_input_dim_per_variable = rnn_input_dim_per_variable
        self.max_context_dim = max_context_dim
        self.task_num = task_num
        self.residual = residual
        self.logits_clip = logits_clip
        self.scale_lb = scale_lb

        # context model
        self.context_model = ContextModel(max_context_dim=max_context_dim, task_num=task_num)

        # causal mask
        self.causal_mask = CausalMask(
            observed_input_dim=self.variable_num + self.action_dim,
            context_input_dim=self.max_context_dim,
            mask_output_dim=self.variable_num,
            logits_clip=self.logits_clip
        )
        mask_dim_list = []
        for i in range(self.variable_num):
            mask_dim_list.extend([i] * self.state_dim_per_variable)
        mask_dim_list.extend(torch.arange(self.variable_num, self.causal_mask.mask_input_dim).tolist())
        self.mask_dim_map = torch.Tensor(mask_dim_list).long()

        # action state projector
        self.action_state_projector = build_parallel_layers(
            input_dim=self.action_dim + self.total_state_dim + self.max_context_dim,
            output_dim=self.rnn_input_dim_per_variable,
            extra_dims=[self.variable_num],
            hidden_dims=[],
            last_activate_name="ELU",
        )

        # rnn
        self.rnn = ParallelGRUCell(
            input_size=self.rnn_input_dim_per_variable,
            hidden_size=self.hidden_dim_per_variable,
            extra_dims=[self.variable_num],
        )

        # rnn to prior projector
        self.rnn_to_prior_projector = NormalParamWrapper(
            build_parallel_layers(
                input_dim=self.hidden_dim_per_variable,
                output_dim=self.state_dim_per_variable * 2,
                extra_dims=[self.variable_num],
                hidden_dims=[self.hidden_dim_per_variable],
                activate_name="ELU",
            ),
            scale_lb=scale_lb,
            scale_mapping="softplus",
        )

    def get_parameter(self, target: str):
        if target == "module":
            return chain(
                self.action_state_projector.parameters(),
                self.rnn.parameters(),
                self.rnn_to_prior_projector.parameters(),
            )
        elif target == "mask_logits":
            return self.causal_mask.parameters()
        elif target == "context_hat":
            return self.context_model.parameters()
        else:
            raise NotImplementedError

    @property
    def total_state_dim(self):
        return self.state_dim_per_variable * self.variable_num

    @property
    def total_hidden_dim(self):
        return self.hidden_dim_per_variable * self.variable_num

    def forward(self, state, belief, action, idx=None, deterministic_mask=False):
        """get prior distribution and next state from current state, belief and action, input data should be 1-d or 2-d,
            because time dimension is dealt in ``RSSMRollout`` class
        """

        assert len(state.shape) == len(belief.shape) == len(action.shape) < 3, "state should be 1-d or 2-d"
        one_dim = len(state.shape) == 1  # single state, used by policy

        if one_dim:
            state, belief, action = state.unsqueeze(0), belief.unsqueeze(0), action.unsqueeze(0)
            if idx is not None:
                idx = idx.unsqueeze(0)

        batch_size, state_dim = state.shape

        if idx is None:
            idx = torch.empty(batch_size, 1).long().to(state.device)

        projector_input = torch.cat([state, action, self.context_model(idx)], dim=-1)
        masked_projector_input, mask = self.causal_mask(
            inputs=projector_input,
            dim_map=self.mask_dim_map,
            deterministic=deterministic_mask
        )
        masked_action_state = self.action_state_projector(masked_projector_input)
        reshaped_belief = belief.reshape(-1, self.variable_num, self.hidden_dim_per_variable).permute(1, 0, 2)

        next_belief = self.rnn(masked_action_state, reshaped_belief)

        prior_mean, prior_std = self.rnn_to_prior_projector(next_belief)
        if self.residual:
            reshaped_state = state.reshape(-1, self.variable_num, self.state_dim_per_variable).permute(1, 0, 2)
            prior_mean = prior_mean + reshaped_state
        prior_mean = prior_mean.permute(1, 0, 2).reshape(batch_size, -1)
        prior_std = prior_std.permute(1, 0, 2).reshape(batch_size, -1)

        next_state = prior_mean + torch.randn_like(prior_std) * prior_std
        next_belief = next_belief.permute(1, 0, 2).reshape(batch_size, -1)

        if one_dim:
            prior_mean, prior_std, next_state, next_belief = prior_mean.squeeze(0), prior_std.squeeze(0), \
                next_state.squeeze(0), next_belief.squeeze(0)

        return prior_mean, prior_std, next_state, next_belief


def test_causal_rssm_prior():
    action_dim = 2
    variable_num = 10
    state_dim_per_variable = 3
    hidden_dim_per_variable = 20
    max_context_dim = 10
    task_num = 100
    batch_size = 32

    prior = CausalRSSMPrior(
        action_dim=action_dim,
        variable_num=variable_num,
        state_dim_per_variable=state_dim_per_variable,
        hidden_dim_per_variable=hidden_dim_per_variable,
        max_context_dim=max_context_dim,
        task_num=task_num,
    )

    state = torch.randn(batch_size, variable_num * state_dim_per_variable)
    belief = torch.randn(batch_size, variable_num * hidden_dim_per_variable)
    action = torch.randn(batch_size, action_dim)
    idx = torch.randint(0, task_num, (batch_size, 1))

    prior_mean, prior_std, next_state, next_belief = prior(state, belief, action, idx)

    assert prior_mean.shape == prior_std.shape == next_state.shape == state.shape
    assert next_belief.shape == belief.shape


if __name__ == '__main__':
    test_causal_rssm_prior()
