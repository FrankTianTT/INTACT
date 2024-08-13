# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch


def ones(
        obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor
) -> torch.Tensor:
    return torch.ones(*next_obs.shape[:-1], 1).to(next_obs.device)


def heating(
        obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor
) -> torch.Tensor:
    temp = next_obs * 10 + 20
    return -(temp - 20).abs().sum(dim=-1, keepdim=True)


def cartpole_swingup(
        obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor
) -> torch.Tensor:
    theta = next_obs[..., 2]
    return (torch.cos(theta) + 1) / 2


# def half_cheetah(
#         obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor
# ) -> torch.Tensor:
#     theta = next_obs[..., 2]
#     return (torch.cos(theta) + 1) / 2


reward_fns_dict = {
    "ones": ones,
    "heating": heating,
    "cartpole_swingup": cartpole_swingup
}
