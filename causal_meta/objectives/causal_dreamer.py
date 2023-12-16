import torch
from tensordict import TensorDict

from torchrl.objectives.dreamer import DreamerModelLoss
from causal_meta.modules.tensordict_module.causal_dreamer_wrapper import CausalDreamerWrapper


class CausalDreamerModelLoss(DreamerModelLoss):

    def __init__(
            self,
            world_model: CausalDreamerWrapper,
            sparse_weight: float = 0.02,
            context_sparse_weight: float = 0.01,
            context_max_weight: float = 0.2,
            sampling_times: int = 30,
            **kwargs
    ):
        super().__init__(world_model, **kwargs)
        self.sparse_weight = sparse_weight
        self.context_sparse_weight = context_sparse_weight
        self.context_max_weight = context_max_weight
        self.sampling_times = sampling_times

        self.variable_num = self.world_model.variable_num
        self.state_dim_per_variable = self.world_model.state_dim_per_variable

        self.causal_mask = self.world_model.causal_mask

    def reinforce(self, tensordict: TensorDict) -> torch.Tensor:
        tensordict = tensordict.clone(recurse=False)
        mask = tensordict.get(self.tensor_keys.collector_mask).clone()

        tensordict = self.world_model.parallel_forward(tensordict, self.sampling_times)

        kl_loss = self.kl(
            tensordict.get(("next", self.tensor_keys.prior_mean))[:, mask],
            tensordict.get(("next", self.tensor_keys.prior_std))[:, mask],
            tensordict.get(("next", self.tensor_keys.posterior_mean))[:, mask],
            tensordict.get(("next", self.tensor_keys.posterior_std))[:, mask],
        )

        sampling_loss = kl_loss.reshape(*kl_loss.shape[:-1], self.variable_num, self.state_dim_per_variable)
        sampling_loss = sampling_loss.sum(dim=-1)
        sampling_mask = tensordict.get("causal_mask")[:, mask]

        mask_grad = self.causal_mask.total_mask_grad(
            sampling_mask=sampling_mask,
            sampling_loss=sampling_loss,
            sparse_weight=self.sparse_weight,
            context_sparse_weight=self.context_sparse_weight,
            context_max_weight=self.context_max_weight
        )

        return mask_grad

    def kl(
            self,
            prior_mean: torch.Tensor,
            prior_std: torch.Tensor,
            posterior_mean: torch.Tensor,
            posterior_std: torch.Tensor,
    ) -> torch.Tensor:
        kl = (
                torch.log(prior_std / posterior_std)
                + (posterior_std ** 2 + (prior_mean - posterior_mean) ** 2)
                / (2 * prior_std ** 2)
                - 0.5
        )
        return kl
