import torch
from tensordict import TensorDict
from torch.nn.functional import binary_cross_entropy_with_logits
from torchrl.objectives.utils import distance_loss
from torchrl.objectives.dreamer import DreamerModelLoss

from intact.modules.tensordict_module.dreamer_wrapper import DreamerWrapper


class CausalDreamerModelLoss(DreamerModelLoss):
    def __init__(
        self,
        world_model: DreamerWrapper,
        sparse_weight: float = 0.02,
        context_sparse_weight: float = 0.01,
        context_max_weight: float = 0.2,
        sampling_times: int = 30,
        free_nats: float = 0.5,
        global_average: bool = False,
        delayed_clamp: bool = False,
        **kwargs
    ):
        self.model_type = world_model.model_type
        assert not global_average, "global_average is not supported in CausalDreamerModelLoss"
        assert not delayed_clamp, "delayed_clamp is not supported in CausalDreamerModelLoss"
        super().__init__(world_model, free_nats=free_nats, global_average=False, delayed_clamp=False, **kwargs)

        self.sparse_weight = sparse_weight
        self.context_sparse_weight = context_sparse_weight
        self.context_max_weight = context_max_weight
        self.sampling_times = sampling_times

        self.variable_num = self.world_model.variable_num
        self.state_dim_per_variable = self.world_model.state_dim_per_variable

        if self.model_type == "causal":
            self.causal_mask = self.world_model.causal_mask
            self.using_reinforce = self.causal_mask.reinforce
        else:
            self.causal_mask, self.using_reinforce = None, None

    def forward(self, tensordict: TensorDict):
        model_loss_td, sampled_tensordict = super().forward(tensordict)
        if self.model_type == "causal" and not self.using_reinforce:
            model_loss_td.set("sparse_loss", torch.sigmoid(self.causal_mask.observed_logits).sum() * self.sparse_weight)
            if self.causal_mask.context_input_dim > 0:
                model_loss_td.set(
                    "context_sparse_loss", torch.sigmoid(self.causal_mask.context_logits).sum() * self.context_sparse_weight
                )
                model_loss_td.set(
                    "context_max_loss",
                    torch.sigmoid(self.causal_mask.context_logits).max(dim=1).sum() * self.context_max_weight,
                )
        return model_loss_td, sampled_tensordict

    def reinforce(self, tensordict: TensorDict):
        tensordict = tensordict.clone(recurse=False)
        mask = tensordict.get(self.tensor_keys.collector_mask).clone()

        tensordict = self.world_model.parallel_forward(tensordict, self.sampling_times)

        sampling_loss = self.kl_loss(
            tensordict.get(("next", self.tensor_keys.prior_mean))[:, mask],
            tensordict.get(("next", self.tensor_keys.prior_std))[:, mask],
            tensordict.get(("next", self.tensor_keys.posterior_mean))[:, mask],
            tensordict.get(("next", self.tensor_keys.posterior_std))[:, mask],
            keep_dim=True,
        )  # (sampling_times, batch_size, variable_num, state_dim_per_variable)
        sampling_loss *= self.lambda_kl
        sampling_mask = tensordict.get("causal_mask")[:, mask]

        mask_grad = self.causal_mask.total_mask_grad(
            sampling_mask=sampling_mask,
            sampling_loss=sampling_loss,
            sparse_weight=self.sparse_weight,
            context_sparse_weight=self.context_sparse_weight,
            context_max_weight=self.context_max_weight,
        )

        return mask_grad, sampling_loss

    def kl_loss(
        self,
        prior_mean: torch.Tensor,
        prior_std: torch.Tensor,
        posterior_mean: torch.Tensor,
        posterior_std: torch.Tensor,
        keep_dim: bool = False,
    ) -> torch.Tensor:
        kl = (
            torch.log(prior_std / posterior_std)
            + (posterior_std**2 + (prior_mean - posterior_mean) ** 2) / (2 * prior_std**2)
            - 0.5
        )
        if self.model_type == "causal":
            kl = kl.reshape(*kl.shape[:-1], self.variable_num, self.state_dim_per_variable)
            kl = kl.sum(-1)

            free_nats_every_variable = self.free_nats / self.variable_num
            kl = kl.clamp_min(free_nats_every_variable)

            if keep_dim:
                return kl
            else:
                return kl.sum(-1).mean()
        else:
            kl = kl.sum(-1)
            return kl.clamp_min(self.free_nats).mean()
