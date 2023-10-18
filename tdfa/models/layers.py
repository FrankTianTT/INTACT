from typing import List, Optional, Tuple
from itertools import product

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def truncated_normal_(tensor: torch.Tensor, mean: float = 0, std: float = 1) -> torch.Tensor:
    """Samples from a truncated normal distribution in-place.

    Args:
        tensor (tensor): the tensor in which sampled values will be stored.
        mean (float): the desired mean (default = 0).
        std (float): the desired standard deviation (default = 1).

    Returns:
        (tensor): the tensor with the stored values. Note that this modifies the input tensor
            in place, so this is just a pointer to the same object.
    """
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    while True:
        cond = torch.logical_or(tensor < mean - 2 * std, tensor > mean + 2 * std)
        bound_violations = torch.sum(cond).item()
        if bound_violations == 0:
            break
        tensor[cond] = torch.normal(mean, std, size=(bound_violations,), device=tensor.device)
    return tensor


class ParallelLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            extra_dims: Optional[List[int]] = None,
            bias: bool = True,
            init_type: str = "truncated_normal",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.extra_dims = [] if extra_dims is None else extra_dims
        self.init_type = init_type

        self.weight = nn.Parameter(torch.zeros(*self.extra_dims, self.in_features, self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(*self.extra_dims, 1, self.out_features))
            self.use_bias = True
        else:
            self.use_bias = False

        self.init_params()

    def init_params(self):
        """Initialize weights and biases. Currently, only `kaiming_uniform` and `truncated_normal` are supported.

        Returns: None

        """
        if self.init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        elif self.init_type == "truncated_normal":
            stddev = 1 / (2 * np.sqrt(self.in_features))
            for dims in product(*map(range, self.extra_dims)):
                truncated_normal_(self.weight.data[dims], std=stddev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xw = x.matmul(self.weight)
        if self.use_bias:
            return xw + self.bias
        else:
            return xw

    def extra_repr(self):
        return 'input_dims={}, output_dims={}, extra_dims={}, bias={}, init_type="{}"'.format(
            self.in_features, self.out_features, str(self.extra_dims), self.use_bias, self.init_type
        )
