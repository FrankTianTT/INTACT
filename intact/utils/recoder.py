# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import pathlib
import warnings
from collections import defaultdict, OrderedDict
from copy import deepcopy
from textwrap import indent
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch.nn
from tensordict.nn import TensorDictModule
from tensordict.tensordict import pad, TensorDictBase
from torchrl.envs.common import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.trainers import Recorder as BaseRecorder


class Recorder(BaseRecorder):
    def __init__(
        self,
        *,
        record_interval: int,
        env_max_steps: int = 1000,
        eval_repeat_times: int = 3,
        frame_skip: int = 1,
        policy_exploration: TensorDictModule = ExplorationType.MODE,
        environment: EnvBase = None,
        exploration_type: ExplorationType = ExplorationType.RANDOM,
        log_keys: Optional[List[Union[str, Tuple[str]]]] = None,
        out_keys: Optional[Dict[Union[str, Tuple[str]], str]] = None,
        suffix: Optional[str] = None,
        log_pbar: bool = False,
        recorder: EnvBase = None,
    ) -> None:
        self.env_max_steps = env_max_steps
        self.eval_repeat_times = eval_repeat_times
        super().__init__(
            record_interval=record_interval,
            record_frames=env_max_steps * eval_repeat_times,
            frame_skip=frame_skip,
            policy_exploration=policy_exploration,
            environment=environment,
            exploration_type=exploration_type,
            log_keys=log_keys,
            out_keys=out_keys,
            suffix=suffix,
            log_pbar=log_pbar,
            recorder=recorder,
        )

        assert self.log_keys == [("next", "reward")]

    @torch.inference_mode()
    def __call__(self, batch: TensorDictBase) -> Dict:
        out = None
        if self._count % self.record_interval == 0:
            with set_exploration_type(self.exploration_type):
                if isinstance(self.policy_exploration, torch.nn.Module):
                    self.policy_exploration.eval()
                self.environment.eval()

                out = {"r_evaluation": 0, "total_r_evaluation": 0}
                for i in range(self.eval_repeat_times):
                    td_record = self.environment.rollout(
                        policy=self.policy_exploration,
                        max_steps=self.env_max_steps,
                        auto_reset=True,
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                    ).clone()

                    rewards = td_record[("next", "reward")].float()
                    out["r_evaluation"] += rewards.mean() / self.frame_skip
                    out["total_r_evaluation"] += rewards.sum(dim=td_record.ndim - 1).mean()
                out["r_evaluation"] /= self.eval_repeat_times
                out["total_r_evaluation"] /= self.eval_repeat_times

                if isinstance(self.policy_exploration, torch.nn.Module):
                    self.policy_exploration.train()
                self.environment.train()
                if hasattr(self.environment, "transform"):
                    self.environment.transform.dump(suffix=self.suffix)

        self._count += 1
        self.environment.close()
        return out

    def state_dict(self) -> Dict:
        return {
            "_count": self._count,
            "recorder_state_dict": self.environment.state_dict(),
        }
