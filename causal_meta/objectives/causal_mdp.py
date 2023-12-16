from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torchrl.objectives.common import LossModule
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from causal_meta.modules.utils import build_mlp
from causal_meta.stats.metric import mutual_info_estimation
from causal_meta.modules.tensordict_module.mdp_wrapper import MDPWrapper


class CausalWorldModelLoss(LossModule):
    def __init__(
            self,
            world_model: MDPWrapper,
            lambda_transition: float = 1.0,
            lambda_reward: float = 1.0,
            lambda_terminated: float = 1.0,
            lambda_mutual_info: float = 1.0,  # use for meta identify
            sparse_weight: float = 0.05,
            context_sparse_weight: float = 0.01,
            context_max_weight: float = 0.1,
            sampling_times: int = 50,
    ):
        super().__init__()
        self.world_model = world_model
        self.model_type = world_model.model_type

        if self.model_type == "causal":
            self.causal_mask = world_model.causal_mask
        else:
            self.causal_mask = None
        self.context_model = world_model.context_model
        self.learn_obs_var = world_model.learn_obs_var

        self.lambda_transition = lambda_transition
        self.lambda_reward = lambda_reward
        self.lambda_terminated = lambda_terminated
        self.lambda_mutual_info = lambda_mutual_info
        self.sparse_weight = sparse_weight
        self.context_sparse_weight = context_sparse_weight
        self.context_max_weight = context_max_weight
        self.sampling_times = sampling_times

    def loss(self, tensordict, reduction="none"):
        assert len(tensordict.batch_size) == 1

        if self.learn_obs_var:
            transition_loss = F.gaussian_nll_loss(
                tensordict.get("obs_mean"),
                tensordict.get(("next", "observation")),
                torch.exp(tensordict.get("obs_log_var")),
                reduction=reduction
            )
        else:
            transition_loss = F.mse_loss(
                tensordict.get("obs_mean"),
                tensordict.get(("next", "observation")),
                reduction=reduction
            )

        reward_loss = F.mse_loss(
            tensordict.get("reward"),
            tensordict.get(("next", "reward")),
            reduction=reduction
        )
        terminated_loss = F.binary_cross_entropy_with_logits(
            tensordict.get("terminated"),
            tensordict.get(("next", "terminated")).float(),
            reduction=reduction
        )

        loss_td = TensorDict({
            "transition_loss": transition_loss,
            "reward_loss": reward_loss,
            "terminated_loss": terminated_loss,
        },
            batch_size=transition_loss.shape[0]
        )

        loss_tensor = torch.cat([
            transition_loss * self.lambda_transition,
            reward_loss * self.lambda_reward,
            terminated_loss * self.lambda_terminated
        ], dim=1)

        return loss_td, loss_tensor

    def forward(self, tensordict: TensorDict, deterministic_mask=False):
        tensordict = tensordict.clone(recurse=False)

        if self.model_type == "causal":
            tensordict = self.world_model(tensordict, deterministic_mask=deterministic_mask)
        else:
            tensordict = self.world_model(tensordict)

        loss_td, loss_tensor = self.loss(tensordict)
        if self.lambda_mutual_info > 0:
            if self.model_type == "causal":
                valid_context_idx = self.causal_mask.valid_context_idx
            else:
                valid_context_idx = torch.arange(self.context_model.max_context_dim)
            mutual_info_loss = self.context_model.get_mutual_info(
                idx=tensordict["idx"],
                valid_context_idx=valid_context_idx,
                reduction="none"
            ).reshape(-1, 1)
            loss_td.set("mutual_info_loss", mutual_info_loss)
            loss_tensor = torch.cat([
                loss_tensor,
                mutual_info_loss * self.lambda_mutual_info
            ], dim=-1)

        if self.model_type == "inn":
            tensordict = self.world_model.inv_forward(tensordict)

            gt_context = self.context_model(tensordict["idx"])
            context_loss = 0.5 * (tensordict["inv_context"] - gt_context) ** 2
            loss_td.set("context_loss", context_loss)
            # TODO: cat loss_tensor
        return loss_td, loss_tensor.mean()

    def reinforce(self, tensordict: TensorDict):
        assert self.model_type == "causal", "reinforce is only available for CausalWorldModel"
        assert self.causal_mask.reinforce, "causal_mask should be learned by reinforce"

        tensordict = tensordict.clone()
        tensordict = self.world_model.parallel_forward(tensordict, self.sampling_times)

        _, loss_tensor = self.loss(tensordict.reshape(-1), reduction="none")
        sampling_loss = loss_tensor.reshape(*tensordict.batch_size, -1)

        mask_grad = self.causal_mask.total_mask_grad(
            sampling_mask=tensordict.get("causal_mask"),
            sampling_loss=sampling_loss,
            sparse_weight=self.sparse_weight,
            context_sparse_weight=self.context_sparse_weight,
            context_max_weight=self.context_max_weight
        )

        return mask_grad


def test_causal_world_model_loss():
    from causal_meta.modules.models.mdp_world_model import CausalWorldModel
    from causal_meta.modules.tensordict_module.mdp_wrapper import MDPWrapper

    obs_dim = 4
    action_dim = 1
    max_context_dim = 10
    task_num = 100
    batch_size = 32

    world_model = CausalWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        meta=True,
        max_context_dim=max_context_dim,
        task_num=task_num,
    )
    causal_mdp_wrapper = MDPWrapper(world_model)
    mdp_loss = CausalWorldModelLoss(causal_mdp_wrapper)

    td = TensorDict({
        "observation": torch.randn(batch_size, obs_dim),
        "action": torch.randn(batch_size, action_dim),
        "idx": torch.randint(0, task_num, (batch_size, 1)),
        "next": {
            "terminated": torch.randn(batch_size, 1) > 0,
            "reward": torch.randn(batch_size, 1),
            "observation": torch.randn(batch_size, obs_dim),
        }
    },
        batch_size=batch_size,
    )

    td = causal_mdp_wrapper(td)
    loss_td, total_loss = mdp_loss(td)

    print(loss_td)


def test_inn_world_model_loss():
    from causal_meta.modules.models.mdp_world_model import INNWorldModel
    from causal_meta.modules.tensordict_module.mdp_wrapper import MDPWrapper

    obs_dim = 4
    action_dim = 1
    task_num = 100
    batch_size = 32

    world_model = INNWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        task_num=task_num,
    )
    inn_mdp_wrapper = MDPWrapper(world_model)
    mdp_loss = CausalWorldModelLoss(inn_mdp_wrapper)

    td = TensorDict({
        "observation": torch.randn(batch_size, obs_dim),
        "action": torch.randn(batch_size, action_dim),
        "idx": torch.randint(0, task_num, (batch_size, 1)),
        "next": {
            "terminated": torch.randn(batch_size, 1) > 0,
            "reward": torch.randn(batch_size, 1),
            "observation": torch.randn(batch_size, obs_dim),
        }
    },
        batch_size=batch_size,
    )

    loss = mdp_loss(td)


if __name__ == '__main__':
    test_causal_world_model_loss()
