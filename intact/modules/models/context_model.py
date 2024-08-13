import torch
from torch import nn

from intact.stats.mcc import mean_corr_coef
from intact.stats.metric import mutual_info_estimation


class ContextModel(nn.Module):
    def __init__(
        self,
        meta=False,
        max_context_dim=0,
        task_num=0,
        init_scale=0.0,
        context_clip=1.,
    ):
        super().__init__()
        self.meta = meta
        self.max_context_dim = max_context_dim
        self.task_num = task_num
        self.init_scale = init_scale
        self.context_clip = context_clip

        init_context_hat = torch.randn(task_num, max_context_dim) * init_scale
        self._context_hat: torch.nn.Parameter = torch.nn.Parameter(
            init_context_hat
        )

        self.fixed_idx = None

    @property
    def context_dim(self):
        return self.max_context_dim if self.meta else 0

    @property
    def device(self):
        return self._context_hat.device

    def set_context(self, context):
        assert context.shape == self._context_hat.shape
        if self.context_clip > 0:
            context = torch.clamp(
                context, -self.context_clip, self.context_clip
            )
        self._context_hat.data = context.to(self.device)

    def fix(self, idx=None):
        assert self.meta
        if idx is None:
            self.fixed_idx = torch.arange(self.max_context_dim).to(self.device)
        else:
            self.fixed_idx = torch.tensor(idx).to(int).to(self.device)

    def unfix(self):
        self.fixed_idx = None

    @property
    def context_hat(self):
        if self.fixed_idx is None:
            context_hat = self._context_hat
        else:
            context_hat = torch.zeros_like(self._context_hat).to(self.device)
            context_hat[:, self.fixed_idx] += self._context_hat.detach()[
                :, self.fixed_idx
            ]
            unfixed_idx = ~torch.isin(
                torch.arange(self.max_context_dim).to(self.device),
                self.fixed_idx,
            )
            context_hat[:, unfixed_idx] += self._context_hat[:, unfixed_idx]
        return torch.clamp(context_hat, -self.context_clip, self.context_clip)

    def reset(self, task_num=None):
        self.task_num = task_num or self.task_num

        init_context_hat = (
            torch.randn(self.task_num, self.max_context_dim) * self.init_scale
        )
        self._context_hat.data = init_context_hat.to(self.device)

    def extra_repr(self):
        if self.meta:
            return "max_context_dim={}, task_num={}".format(
                self.max_context_dim,
                self.task_num,
            )
        else:
            return ""

    def forward(self, idx=None):
        if idx is None:
            assert not self.meta, "idx should not be None when meta is True"
            return torch.empty(0).to(self.device)

        assert (
            idx.shape[-1] == 1
        ), f"last dim of idx should be 1, got {idx.shape}"
        return self.context_hat[idx[..., 0]]

    def get_mutual_info(self, idx, valid_context_idx=None, reduction="mean"):
        context_hat = self(idx)
        if valid_context_idx is not None:
            context_hat = context_hat[:, valid_context_idx]
        return mutual_info_estimation(context_hat, reduction=reduction)

    def get_mcc(
        self,
        context_gt,
        valid_idx=None,
        return_permutation=True,
        method="pearson",
    ):
        if isinstance(context_gt, torch.Tensor):
            context_gt = context_gt.detach().cpu().numpy()

        if valid_idx is None:
            context_hat = self.context_hat.detach().cpu().numpy()
        else:
            context_hat = self.context_hat[:, valid_idx].detach().cpu().numpy()

        mcc, permutation = mean_corr_coef(
            context_hat, context_gt, return_permutation=True, method=method
        )

        if return_permutation:
            return mcc, permutation, context_hat
        else:
            return mcc
