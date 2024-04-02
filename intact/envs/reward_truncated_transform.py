from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Tuple, Union
from copy import copy

import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import expand_as_right, NestedKey
from torchrl.envs import Transform, RewardSum, Compose
from torchrl.data.tensor_specs import TensorSpec, CompositeSpec, DiscreteTensorSpec


class RewardTruncatedTransform(Transform):
    def __init__(
        self,
        in_keys: Sequence[NestedKey] = None,
        out_keys: Sequence[NestedKey] = None,
    ):
        self.reset_key = "_reset"
        self.ever_reset: Optional[torch.Tensor] = None

        if in_keys is None:
            in_keys = "reward"
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def _apply_transform(self, reward: torch.Tensor) -> torch.Tensor:
        reward = torch.where(self.ever_reset.reshape(reward.shape), torch.zeros_like(reward), reward)
        return reward

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        if self.reset_key not in tensordict.keys():  # external reset call
            self.ever_reset = torch.zeros(tensordict.batch_size, dtype=torch.bool, device=tensordict.device)
        else:
            self.ever_reset = torch.logical_or(self.ever_reset, tensordict[self.reset_key])
        return tensordict_reset
