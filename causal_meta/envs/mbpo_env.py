# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union, List

import numpy as np
import torch

from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.model_based import ModelBasedEnvBase
from tensordict import TensorDict
from tensordict.nn import TensorDictModule


class MBPOEnv(ModelBasedEnvBase):
    def __init__(
            self,
            world_model: TensorDictModule,
            num_networks: int,
            device="cpu",
            dtype=None,
            batch_size=None,
    ):
        super(MBPOEnv, self).__init__(
            world_model=world_model,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        self.num_networks = num_networks
        self.elites = torch.arange(self.num_networks)

    def _step(self, tensordict: TensorDict) -> TensorDict:
        tensordict_out = tensordict.clone()
        # Compute world state
        sampled_model_id = self.elites[
            torch.randint(0, len(self.elites), tensordict_out.shape)
        ]

        tensordict_out = self.world_model(tensordict_out)
        tensordict_out = tensordict_out[
            sampled_model_id, torch.arange(tensordict_out.shape[1])
        ]
        # Step requires a done flag. No sense for MBRL so we set it to False
        if "done" not in self.world_model.out_keys:
            tensordict_out["done"] = torch.zeros(tensordict_out.shape, dtype=torch.bool)
        return tensordict_out

    def _reset(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        batch_size = tensordict.batch_size if tensordict is not None else []
        device = tensordict.device if tensordict is not None else self.device
        tensordict = TensorDict(
            {},
            batch_size=batch_size,
            device=device,
        )
        tensordict = tensordict.update(self.state_spec.rand(batch_size))
        tensordict = tensordict.update(self.observation_spec.rand(batch_size))

        return tensordict

    def _set_seed(self, seed: Optional[int]) -> int:
        return seed
