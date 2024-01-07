# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from causal_meta.envs import termination_fns


def ones(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    return torch.ones(*next_obs.shape[:-1], 1).to(next_obs.device)


def heating(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    return - (next_obs - 20).abs().sum(dim=-1, keepdim=True)


reward_fns_dict = {
    "ones": ones,
    "heating": heating,
}

if __name__ == '__main__':
    obs = torch.rand(10, 4) * 40
    act = torch.randn(10, 1)
    next_obs = torch.rand(10, 4) * 40
    print(next_obs)

    print(heating(obs, act, next_obs))
