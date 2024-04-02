# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch


def hopper(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    height = next_obs[..., 0]
    angle = next_obs[..., 1]
    not_done: torch.Tensor = (
        torch.isfinite(next_obs).all(-1) * (next_obs[..., 1:].abs() < 100).all(-1) * (height > 0.7) * (angle.abs() < 0.2)
    )
    return (~not_done).unsqueeze(dim=-1)


def cartpole(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    x, theta = next_obs[..., 0], next_obs[..., 2]

    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * math.pi / 360
    not_done: torch.Tensor = (
        (x > -x_threshold) * (x < x_threshold) * (theta > -theta_threshold_radians) * (theta < theta_threshold_radians)
    )
    return (~not_done).unsqueeze(dim=-1)


def inverted_pendulum(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    not_done: torch.Tensor = torch.isfinite(next_obs).all(-1) * (next_obs[..., 1].abs() <= 0.2)
    return (~not_done).unsqueeze(dim=-1)


def no_termination(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    return torch.zeros(*next_obs.shape[:-1], 1).bool().to(next_obs.device)


def walker2d(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    height = next_obs[..., 0]
    angle = next_obs[..., 1]
    not_done: torch.Tensor = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
    return (~not_done).unsqueeze(dim=-1)


termination_fns_dict = {
    "cartpole": cartpole,
    "inverted_pendulum": inverted_pendulum,
    "hopper": hopper,
    "no_termination": no_termination,
    "walker2d": walker2d,
}


def test_termination_fn():
    for fn in termination_fns_dict.values():
        print(f"testing {fn.__name__} ...")
        for batch_size in [(), (10,), (10, 20)]:
            obs = torch.randn(*batch_size, 11)
            act = torch.randn(*batch_size, 3)
            next_obs = torch.randn(*batch_size, 11)
            done = fn(obs, act, next_obs)
            assert done.shape == (*batch_size, 1)
            assert done.dtype == torch.bool
        print("passed")


if __name__ == "__main__":
    test_termination_fn()
