# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.envs.common import EnvBase
from torchrl.envs.utils import step_mdp
from torchrl.modules import CEMPlanner


class MyCEMPlanner(CEMPlanner):
    def __init__(
        self,
        env: EnvBase,
        planning_horizon: int,
        optim_steps: int,
        num_candidates: int,
        top_k: int,
        reward_key: str = ("next", "reward"),
        action_key: str = "action",
        alpha=0.1,
    ):
        super().__init__(
            env=env,
            planning_horizon=planning_horizon,
            optim_steps=optim_steps,
            num_candidates=num_candidates,
            top_k=top_k,
            reward_key=reward_key,
            action_key=action_key,
        )

        self.alpha = alpha

    def planning(self, tensordict: TensorDictBase) -> torch.Tensor:
        batch_size = tensordict.batch_size
        action_shape = (
            *batch_size,
            self.num_candidates,
            self.planning_horizon,
            *self.action_spec.shape,
        )
        action_stats_shape = (
            *batch_size,
            1,
            self.planning_horizon,
            *self.action_spec.shape,
        )
        action_topk_shape = (
            *batch_size,
            self.top_k,
            self.planning_horizon,
            *self.action_spec.shape,
        )
        TIME_DIM = len(self.action_spec.shape) - 3
        K_DIM = len(self.action_spec.shape) - 4
        expanded_original_tensordict = (
            tensordict.unsqueeze(-1)
            .expand(*batch_size, self.num_candidates)
            .to_tensordict()
        )
        _action_means = torch.zeros(
            *action_stats_shape,
            device=tensordict.device,
            dtype=self.env.action_spec.dtype,
        )
        _action_stds = torch.ones_like(_action_means)
        container = TensorDict(
            {
                "tensordict": expanded_original_tensordict,
                "stats": TensorDict(
                    {
                        "_action_means": _action_means,
                        "_action_stds": _action_stds,
                    },
                    [*batch_size, 1, self.planning_horizon],
                ),
            },
            batch_size,
        )

        for _ in range(self.optim_steps):
            actions_means = container.get(("stats", "_action_means"))
            actions_stds = container.get(("stats", "_action_stds"))
            actions = actions_means + actions_stds * torch.randn(
                *action_shape,
                device=actions_means.device,
                dtype=actions_means.dtype,
            )
            actions = self.env.action_spec.project(actions)
            optim_tensordict = container.get("tensordict").clone()
            policy = _PrecomputedActionsSequentialSetter(actions)
            optim_tensordict = self.reward_truncated_rollout(
                policy=policy, tensordict=optim_tensordict
            )

            sum_rewards = optim_tensordict.get(self.reward_key).sum(
                dim=TIME_DIM, keepdim=True
            )
            _, top_k = sum_rewards.topk(self.top_k, dim=K_DIM)
            top_k = top_k.expand(action_topk_shape)
            best_actions = actions.gather(K_DIM, top_k)
            self.update_stats(
                best_actions.mean(dim=K_DIM, keepdim=True),
                best_actions.std(dim=K_DIM, keepdim=True),
                container,
            )
        action_means = container.get(("stats", "_action_means"))
        return action_means[..., 0, 0, :]

    def update_stats(self, means, stds, container):
        self.alpha = 0.1  # should in __init__

        new_means = (
            self.alpha * container.get(("stats", "_action_means"))
            + (1 - self.alpha) * means
        )
        new_stds = (
            self.alpha * container.get(("stats", "_action_stds"))
            + (1 - self.alpha) * stds
        )
        container.set_(("stats", "_action_means"), new_means)
        container.set_(("stats", "_action_stds"), new_stds)

    def reward_truncated_rollout(self, policy, tensordict):
        tensordicts = []
        ever_done = torch.zeros(*tensordict.batch_size, 1, dtype=bool).to(
            self.device
        )
        with torch.no_grad():
            for i in range(self.planning_horizon):
                tensordict = policy(tensordict)
                tensordict = self.env.step(tensordict)
                next_tensordict = step_mdp(tensordict, exclude_action=False)

                tensordict.get(("next", "reward"))[ever_done] = 0
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


class _PrecomputedActionsSequentialSetter:
    def __init__(self, actions):
        self.actions = actions
        self.cmpt = 0

    def __call__(self, tensordict):
        # checks that the step count is lower or equal to the horizon
        if self.cmpt >= self.actions.shape[-2]:
            raise ValueError("Precomputed actions sequence is too short")
        tensordict = tensordict.set("action", self.actions[..., self.cmpt, :])
        self.cmpt += 1
        return tensordict
