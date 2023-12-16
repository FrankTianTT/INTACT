# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch


def hopper(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (
            torch.isfinite(next_obs).all(-1)
            * (next_obs[:, 1:].abs() < 100).all(-1)
            * (height > 0.7)
            * (angle.abs() < 0.2)
    )

    done = ~not_done
    done = done[:, None]
    return done


def cartpole(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    x, theta = next_obs[..., 0], next_obs[..., 2]

    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * math.pi / 360
    not_done = (
            (x > -x_threshold)
            * (x < x_threshold)
            * (theta > -theta_threshold_radians)
            * (theta < theta_threshold_radians)
    )
    done = ~not_done
    done = done.unsqueeze(dim=-1)
    return done


def inverted_pendulum(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    not_done = torch.isfinite(next_obs).all(-1) * (next_obs[..., 1].abs() <= 0.2)
    done = ~not_done
    done = done.unsqueeze(dim=-1)
    return done


def no_termination(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    return torch.zeros(*next_obs.shape[:-1], 1).bool().to(next_obs.device)


#
#
# def walker2d(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
#     assert len(next_obs.shape) == 2
#
#     height = next_obs[:, 0]
#     angle = next_obs[:, 1]
#     not_done = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
#     done = ~not_done
#     done = done[:, None]
#     return done
#
#
# def ant(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor):
#     assert len(next_obs.shape) == 2
#
#     x = next_obs[:, 0]
#     not_done = torch.isfinite(next_obs).all(-1) * (x >= 0.2) * (x <= 1.0)
#
#     done = ~not_done
#     done = done[:, None]
#     return done
#
#
# def humanoid(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor):
#     assert len(next_obs.shape) == 2
#
#     z = next_obs[:, 0]
#     done = (z < 1.0) + (z > 2.0)
#
#     done = done[:, None]
#     return done

termination_fns_dict = {
    "cartpole": cartpole,
    "inverted_pendulum": inverted_pendulum,
    "no_termination": no_termination,
    # "walker2d": termination_fns.walker2d,
    # "ant": termination_fns.ant,
}
