from typing import Optional

import torch
from torch import nn
from torchrl.modules.models.models import MLP

from intact.modules.models.context_model import ContextModel


class Critic(nn.Module):
    def __init__(
        self,
        state_or_obs_dim: int,
        context_dim: int,
        is_mdp=True,
        belief_dim=None,
        depth=3,
        num_cells=200,
        activation_class=nn.ELU,
    ):
        self.state_or_obs_dim = state_or_obs_dim
        self.context_dim = context_dim
        self.context_model: ContextModel = Optional[None]
        self.is_mdp = is_mdp
        self.belief_dim = belief_dim if not is_mdp else 0

        super().__init__()
        self.backbone = MLP(
            in_features=self.state_or_obs_dim + self.belief_dim + self.context_dim,
            out_features=1,
            depth=depth,
            num_cells=num_cells,
            activation_class=activation_class,
        )

    def set_context_model(self, context_model: ContextModel):
        self.context_model = context_model

    def forward(self, state_or_obs, idx, belief=None):
        assert self.context_model is not None, "context model is not set"

        if belief is None:
            inputs = torch.cat([state_or_obs, self.context_model(idx)], dim=-1)
        else:
            inputs = torch.cat([state_or_obs, belief, self.context_model(idx)], dim=-1)
        value = self.backbone(inputs)
        return value
