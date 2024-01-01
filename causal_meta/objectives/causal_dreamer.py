import torch
from tensordict import TensorDict
from torch.nn.functional import binary_cross_entropy_with_logits
from torchrl.objectives.utils import distance_loss
from torchrl.objectives.dreamer import DreamerModelLoss

from causal_meta.modules.tensordict_module.dreamer_wrapper import DreamerWrapper


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

    def reinforce(self, tensordict: TensorDict):
        tensordict = tensordict.clone(recurse=False)
        mask = tensordict.get(self.tensor_keys.collector_mask).clone()

        tensordict = self.world_model.parallel_forward(tensordict, self.sampling_times)

        sampling_loss = self.kl_loss(
            tensordict.get(("next", self.tensor_keys.prior_mean))[:, mask],
            tensordict.get(("next", self.tensor_keys.prior_std))[:, mask],
            tensordict.get(("next", self.tensor_keys.posterior_mean))[:, mask],
            tensordict.get(("next", self.tensor_keys.posterior_std))[:, mask],
            keep_dim=True
        )  # (sampling_times, batch_size, variable_num, state_dim_per_variable)
        sampling_loss *= self.lambda_kl
        sampling_mask = tensordict.get("causal_mask")[:, mask]

        mask_grad = self.causal_mask.total_mask_grad(
            sampling_mask=sampling_mask,
            sampling_loss=sampling_loss,
            sparse_weight=self.sparse_weight,
            context_sparse_weight=self.context_sparse_weight,
            context_max_weight=self.context_max_weight
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
                + (posterior_std ** 2 + (prior_mean - posterior_mean) ** 2)
                / (2 * prior_std ** 2)
                - 0.5
        )
        kl = kl.reshape(*kl.shape[:-1], self.variable_num, self.state_dim_per_variable)
        kl = kl.sum(-1)

        free_nats_every_variable = self.free_nats / self.variable_num
        kl = kl.clamp_min(free_nats_every_variable)

        if keep_dim:
            return kl
        else:
            return kl.sum(-1).mean()


if __name__ == '__main__':
    prior_mean = torch.tensor([0.0002, 0.0051, 0.0111, 0.0164, 0.0103, 0.0040, -0.0062, 0.0049,
                               0.0028, 0.0079, 0.0109, -0.0021, -0.0008, -0.0026, 0.0105, 0.0002,
                               0.0100, 0.0023, -0.0029, 0.0017, -0.0011, -0.0020, -0.0034, 0.0108,
                               0.0083, 0.0002, -0.0038, 0.0077, 0.0053, -0.0025])
    prior_std = torch.tensor([0.2094, 0.2001, 0.1985, 0.5483, 0.2064, 0.2106, 0.2039, 0.2002, 0.2033,
                              0.2057, 0.2000, 0.2046, 0.2018, 0.2083, 0.2049, 0.1952, 0.1945, 0.2033,
                              0.2009, 0.1995, 0.2003, 0.2016, 0.1981, 0.2058, 0.2030, 0.2080, 0.1974,
                              0.5716, 0.2018, 0.1977])
    posterior_mean = torch.tensor([0.0010, 0.0031, 0.0098, -0.5954, 0.0103, 0.0040, -0.0037, 0.0006,
                                   0.0090, 0.0083, 0.0133, -0.0033, 0.0056, 0.0006, 0.0082, 0.0021,
                                   0.0056, 0.0077, -0.0094, 0.0034, -0.0012, -0.0026, -0.0046, 0.0114,
                                   0.0095, 0.0007, -0.0016, -0.4619, 0.0028, 0.0007])
    posterior_std = torch.tensor([0.2126, 0.2037, 0.2020, 0.2190, 0.2103, 0.2139, 0.2076, 0.2040, 0.2077,
                                  0.2092, 0.2030, 0.2083, 0.2047, 0.2121, 0.2080, 0.1984, 0.1979, 0.2065,
                                  0.2047, 0.2032, 0.2037, 0.2053, 0.2016, 0.2087, 0.2071, 0.2118, 0.2012,
                                  0.3574, 0.2046, 0.2017])

    kl1 = torch.log(prior_std / posterior_std)
    kl2 = (posterior_std ** 2 + (prior_mean - posterior_mean) ** 2) / (2 * prior_std ** 2) - 0.5
    print(kl1.sum(), kl2.sum())
    print(kl1.sum() + kl2.sum())
    print(kl1 + kl2)
    print(kl1)
    print(kl2)
