# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import distance_loss


class DreamCriticLoss(LossModule):
    """Dreamer Value Loss.

    Computes the loss of the dreamer value model. The value loss is computed
    between the predicted value and the lambda target.

    Reference: https://arxiv.org/abs/1912.01603.

    Args:
        value_model (TensorDictModule): the value model.
        value_loss (str, optional): the loss to use for the value loss.
            Default: ``"l2"``.
        discount_loss (bool, optional): if ``True``, the loss is discounted with a
            gamma discount factor. Default: False.
        gamma (float, optional): the gamma discount factor. Default: ``0.99``.

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values

        Attributes:
            value (NestedKey): The input tensordict key where the state value is expected.
                Defaults to ``"state_value"``.
        """

        value: NestedKey = "value"
        discount: NestedKey = "discount"

    default_keys = _AcceptedKeys()

    def __init__(
        self,
        value_model: TensorDictModule,
        value_loss: Optional[str] = None,
        discount_loss: bool = False,  # for consistency with paper
        gamma: int = 0.99,
    ):
        super().__init__()
        self.value_model = value_model
        self.value_loss = value_loss if value_loss is not None else "l2"
        self.gamma = gamma
        self.discount_loss = discount_loss

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    def forward(self, fake_data) -> torch.Tensor:
        lambda_target = fake_data.get("lambda_target")
        if self.discount_loss:
            discount = fake_data.get("discount")
        else:
            discount = torch.ones_like(lambda_target).to(fake_data.device)

        # tensordict_select = fake_data.select(*self.value_model.in_keys)
        self.value_model(fake_data)

        # print("discount", discount[0].reshape(-1))
        # print("value", fake_data.get(self.tensor_keys.value)[0].reshape(-1))
        # print("target", lambda_target[0].reshape(-1))

        value_loss = (
            (
                discount
                * distance_loss(
                    fake_data.get(self.tensor_keys.value),
                    lambda_target,
                    self.value_loss,
                )
            )
            .sum((-1, -2))
            .mean()
        )

        loss_tensordict = TensorDict({"loss_value": value_loss}, [])
        return loss_tensordict, fake_data
