import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey

from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_WARNING,
    default_value_kwargs,
    distance_loss,
    hold_out_net,
    ValueEstimators,
)
from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator
from torchrl.objectives.dreamer import DreamerModelLoss
from tdfa.modules.tensordict_module.world_models import CausalDreamerWrapper


class CausalDreamerModelLoss(DreamerModelLoss):

    def __init__(
            self,
            world_model: CausalDreamerWrapper,
            **kwargs
    ):
        super().__init__(world_model, **kwargs)

    def forward(self, tensordict: TensorDict) -> torch.Tensor:
        tensordict = tensordict.clone(recurse=False)
        mask = tensordict.get(self.tensor_keys.collector_mask).clone()

        tensordict.rename_key_(
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.true_reward),
        )
        tensordict = self.world_model(tensordict)
        # compute model loss
        kl_loss = self.kl_loss(
            tensordict.get(("next", self.tensor_keys.prior_mean))[mask],
            tensordict.get(("next", self.tensor_keys.prior_std))[mask],
            tensordict.get(("next", self.tensor_keys.posterior_mean))[mask],
            tensordict.get(("next", self.tensor_keys.posterior_std))[mask],
        ).unsqueeze(-1)
        reco_loss = distance_loss(
            tensordict.get(("next", self.tensor_keys.pixels))[mask],
            tensordict.get(("next", self.tensor_keys.reco_pixels))[mask],
            self.reco_loss,
        )
        if not self.global_average:
            reco_loss = reco_loss.sum((-3, -2, -1))
        reco_loss = reco_loss.mean().unsqueeze(-1)

        reward_loss = distance_loss(
            tensordict.get(("next", self.tensor_keys.true_reward))[mask],
            tensordict.get(("next", self.tensor_keys.reward))[mask],
            self.reward_loss,
        )
        if not self.global_average:
            reward_loss = reward_loss.squeeze(-1)
        reward_loss = reward_loss.mean().unsqueeze(-1)

        if self.pred_continue:
            continue_loss = binary_cross_entropy_with_logits(
                tensordict.get(("next", self.tensor_keys.pred_continue))[mask],
                1 - tensordict.get(("next", self.tensor_keys.terminated)).float()[mask],
                reduction="none"
            )
        else:
            continue_loss = torch.zeros(*mask.shape, 1, device=tensordict.device)

        if not self.global_average:
            continue_loss = continue_loss.squeeze(-1)
        continue_loss = continue_loss.mean().unsqueeze(-1)

        return (
            TensorDict(
                {
                    "loss_model_kl": self.lambda_kl * kl_loss,
                    "loss_model_reco": self.lambda_reco * reco_loss,
                    "loss_model_reward": self.lambda_reward * reward_loss,
                    "loss_model_continue": self.lambda_continue * continue_loss,
                },
                [],
            ),
            tensordict.detach(),
        )
