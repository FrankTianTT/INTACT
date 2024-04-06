import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.objectives.common import LossModule

from intact.modules.tensordict_module.mdp_wrapper import MDPWrapper


class CausalWorldModelLoss(LossModule):
    def __init__(
        self,
        world_model: MDPWrapper,
        lambda_transition: float = 1.0,
        lambda_reward: float = 1.0,
        lambda_terminated: float = 1.0,
        lambda_mutual_info: float = 0.0,  # use for envs identify
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
            self.using_reinforce = self.causal_mask.using_reinforce
        else:
            self.causal_mask, self.using_reinforce = None, None
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
        mask = tensordict.get(("collector", "mask")).clone()

        if self.learn_obs_var:
            transition_loss = F.gaussian_nll_loss(
                tensordict.get("obs_mean")[mask],
                tensordict.get(("next", "observation"))[mask],
                torch.exp(tensordict.get("obs_log_var"))[mask],
                reduction=reduction,
            )
        else:
            transition_loss = F.mse_loss(
                tensordict.get("obs_mean")[mask],
                tensordict.get(("next", "observation"))[mask],
                reduction=reduction,
            )

        if self.learn_obs_var:
            reward_loss = F.gaussian_nll_loss(
                tensordict.get("reward_mean")[mask],
                tensordict.get(("next", "reward"))[mask],
                torch.exp(tensordict.get("reward_log_var"))[mask],
                reduction=reduction,
            )
        else:
            reward_loss = F.mse_loss(
                tensordict.get("reward_mean")[mask],
                tensordict.get(("next", "reward"))[mask],
                reduction=reduction,
            )
        terminated_loss = F.binary_cross_entropy_with_logits(
            tensordict.get("terminated")[mask],
            tensordict.get(("next", "terminated"))[mask].float(),
            reduction=reduction,
        )

        loss_td = TensorDict(
            {
                "transition_loss": transition_loss.clone(),
                "reward_loss": reward_loss.clone(),
                "terminated_loss": terminated_loss.clone(),
            },
            batch_size=transition_loss.shape[0],
        )

        loss_tensor = torch.cat(
            [
                transition_loss * self.lambda_transition,
                reward_loss * self.lambda_reward,
                terminated_loss * self.lambda_terminated,
            ],
            dim=1,
        )

        return loss_td, loss_tensor

    def rollout_forward(
        self, tensordict: TensorDict, deterministic_mask=False
    ):
        tensordict = tensordict.clone(recurse=False)
        assert len(tensordict.shape) == 2

        tensordict_out = []
        *batch, time_steps = tensordict.shape

        for t in range(time_steps):
            _tensordict = tensordict[..., t]
            if t > 0:
                _tensordict["observation"] = tensordict_out[-1][
                    "obs_mean"
                ].clone()

            if self.model_type == "causal":
                _tensordict = self.world_model(
                    _tensordict, deterministic_mask=deterministic_mask
                )
            else:
                _tensordict = self.world_model(_tensordict)

            tensordict_out.append(_tensordict)
        return torch.stack(
            tensordict_out, tensordict.ndimension() - 1
        ).contiguous()

    def forward(
        self, tensordict: TensorDict, deterministic_mask=False, only_train=None
    ):
        tensordict = self.rollout_forward(tensordict)

        loss_td, loss_tensor = self.loss(tensordict)
        if self.lambda_mutual_info > 0:
            if self.model_type == "causal":
                valid_context_idx = self.causal_mask.valid_context_idx
            else:
                valid_context_idx = torch.arange(
                    self.context_model.max_context_dim
                )
            mutual_info_loss = self.context_model.get_mutual_info(
                idx=tensordict["idx"],
                valid_context_idx=valid_context_idx,
                reduction="none",
            ).reshape(-1, 1)
            loss_td.set("mutual_info_loss", mutual_info_loss)
            loss_tensor = torch.cat(
                [loss_tensor, mutual_info_loss * self.lambda_mutual_info],
                dim=-1,
            )

        if self.model_type == "inn":
            tensordict = self.world_model.inv_forward(tensordict)

            gt_context = self.context_model(tensordict["idx"])
            context_loss = 0.5 * (tensordict["inv_context"] - gt_context) ** 2
            loss_td.set("context_loss", context_loss)
            # TODO: cat loss_tensor

        if only_train is not None and self.model_type == "causal":
            not_train = torch.ones(loss_tensor.shape[-1]).to(bool)
            not_train[only_train] = False
            loss_tensor[..., not_train] = 0

        total_loss = loss_tensor.mean()
        if self.model_type == "causal" and not self.using_reinforce:
            total_loss += (
                torch.sigmoid(self.causal_mask.observed_logits).sum()
                * self.sparse_weight
            )
            if self.causal_mask.context_input_dim > 0:
                total_loss += (
                    torch.sigmoid(self.causal_mask.context_logits).sum()
                    * self.context_sparse_weight
                )
                total_loss += (
                    torch.sigmoid(self.causal_mask.context_logits)
                    .max(dim=1)
                    .sum()
                    * self.context_max_weight
                )
        return loss_td, total_loss

    def reinforce_forward(self, tensordict: TensorDict, only_train=None):
        assert (
            self.model_type == "causal"
        ), "reinforce is only available for CausalWorldModel"
        assert (
            self.causal_mask.using_reinforce
        ), "causal_mask should be learned by reinforce"

        tensordict = tensordict.clone()
        mask = tensordict.get(("collector", "mask")).clone()
        tensordict = tensordict[mask]

        with torch.no_grad():
            tensordict = self.world_model.parallel_forward(
                tensordict, self.sampling_times
            )
            _, loss_tensor = self.loss(
                tensordict.reshape(-1), reduction="none"
            )

            sampling_loss = loss_tensor.reshape(*tensordict.batch_size, -1)

            mask_grad = self.causal_mask.total_mask_grad(
                sampling_mask=tensordict.get("causal_mask"),
                sampling_loss=sampling_loss,
                sparse_weight=self.sparse_weight,
                context_sparse_weight=self.context_sparse_weight,
                context_max_weight=self.context_max_weight,
            )

        if only_train is not None:
            not_train = torch.ones(mask_grad.shape[0]).to(bool)
            not_train[only_train] = False
            mask_grad[not_train] = 0

        return mask_grad
