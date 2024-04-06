# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import warnings
from dataclasses import dataclass
from typing import Tuple

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_WARNING,
    default_value_kwargs,
    hold_out_net,
    ValueEstimators,
)
from torchrl.objectives.value import (
    TD0Estimator,
    TD1Estimator,
    TDLambdaEstimator,
)

from intact.envs.mdp_env import MDPEnv


class DreamActorLoss(LossModule):
    @dataclass
    class _AcceptedKeys:
        obs: NestedKey = "observation"
        reward: NestedKey = "reward"
        value: NestedKey = "value"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TDLambda

    def __init__(
        self,
        actor_model: TensorDictModule,
        value_model: TensorDictModule,
        model_based_env: MDPEnv,
        *,
        imagination_horizon: int = 15,
        discount_loss: bool = False,  # for consistency with paper
        pred_continue: bool = True,
        gamma: int = None,
        lmbda: int = None,
        lambda_entropy: float = 3e-4,
    ):
        super().__init__()
        self.actor_model = actor_model
        self.value_model = value_model
        self.model_based_env = model_based_env
        self.imagination_horizon = imagination_horizon
        self.discount_loss = discount_loss
        self.pred_continue = pred_continue
        self.lambda_entropy = lambda_entropy

        if gamma is not None:
            warnings.warn(
                _GAMMA_LMBDA_DEPREC_WARNING, category=DeprecationWarning
            )
            self.gamma = gamma
        if lmbda is not None:
            warnings.warn(
                _GAMMA_LMBDA_DEPREC_WARNING, category=DeprecationWarning
            )
            self.lmbda = lmbda

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                value=self._tensor_keys.value,
            )

    def rollout(self, tensordict):
        tensordicts = []
        ever_done = torch.zeros(*tensordict.batch_size, 1, dtype=bool).to(
            tensordict.device
        )
        for i in range(self.imagination_horizon):
            tensordict = self.actor_model(tensordict)
            tensordict = self.model_based_env.step(tensordict)
            next_tensordict = step_mdp(tensordict, exclude_action=False)

            entropy = 0.5 * torch.log(
                2 * math.pi * math.e * tensordict["scale"] ** 2
            )
            tensordict.set("entropy", entropy)
            tensordicts.append(tensordict)

            ever_done |= tensordict.get(("next", "done"))
            if ever_done.all():
                break
            else:
                tensordict = next_tensordict
        batch_size = (
            self.batch_size if tensordict is None else tensordict.batch_size
        )
        out_td = torch.stack(tensordicts, len(batch_size)).contiguous()
        out_td.refine_names(..., "time")

        return out_td

    def forward(self, tensordict: TensorDict) -> Tuple[TensorDict, TensorDict]:
        with torch.no_grad():
            mask = tensordict.get(("collector", "mask")).clone()
            tensordict = tensordict[mask]

        with hold_out_net(self.model_based_env), set_exploration_type(
            ExplorationType.RANDOM
        ):
            fake_data = self.rollout(tensordict.clone())

            next_tensordict = step_mdp(
                fake_data,
                keep_other=True,
            )
            with hold_out_net(self.value_model):
                next_tensordict = self.value_model(next_tensordict)
        reward = fake_data.get(("next", self.tensor_keys.reward))
        next_value = next_tensordict.get(self.tensor_keys.value)

        terminated = fake_data.get(("next", self.tensor_keys.terminated))
        lambda_target = self.lambda_target(reward, next_value, terminated)
        fake_data.set("lambda_target", lambda_target)

        actor_target = lambda_target + self.lambda_entropy * fake_data.get(
            "entropy"
        )

        if self.discount_loss:
            gamma = self.value_estimator.gamma.to(tensordict.device)
            discount = (1.0 - terminated.float()) * gamma
            # discount = gamma.expand(lambda_target.shape).clone()
            discount = torch.cat(
                [
                    torch.ones_like(discount[..., :1, :]).to(
                        tensordict.device
                    ),
                    discount[..., :-1, :] * gamma,
                ],
                dim=-2,
            )
            discount = discount.cumprod(dim=-2)
            fake_data.set("discount", discount)
            actor_loss = -(actor_target * discount).sum((-2, -1)).mean()
        else:
            actor_loss = -actor_target.sum((-2, -1)).mean()

        loss_tensordict = TensorDict({"loss_actor": actor_loss}, [])
        return loss_tensordict, fake_data.detach()

    def lambda_target(
        self,
        reward: torch.Tensor,
        value: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        done = terminated.clone()
        input_tensordict = TensorDict(
            {
                ("next", self.tensor_keys.reward): reward,
                ("next", self.tensor_keys.value): value,
                ("next", self.tensor_keys.done): done,
                ("next", self.tensor_keys.terminated): terminated,
            },
            [],
        )
        return self.value_estimator.value_estimate(input_tensordict)

    def make_value_estimator(
        self, value_type: ValueEstimators = None, **hyperparams
    ):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        value_net = None
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        if value_type is ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                **hp,
                value_network=value_net,
            )
        elif value_type is ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                **hp,
                value_network=value_net,
            )
        elif value_type is ValueEstimators.GAE:
            if hasattr(self, "lmbda"):
                hp["lmbda"] = self.lmbda
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            if hasattr(self, "lmbda"):
                hp["lmbda"] = self.lmbda
            self._value_estimator = TDLambdaEstimator(
                **hp,
                value_network=value_net,
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "value": self.tensor_keys.value,
            "value_target": "value_target",
        }
        self._value_estimator.set_keys(**tensor_keys)
