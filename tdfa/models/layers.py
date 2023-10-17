from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ParallelLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            extra_dims: Optional[List[int]] = None,
            bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.extra_dims = [] if extra_dims is None else extra_dims

        self.weight = nn.Parameter(torch.zeros(*self.extra_dims, self.in_features, self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(*self.extra_dims, 1, self.out_features))
            self.use_bias = True
        else:
            self.use_bias = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape, self.weight.shape)
        xw = x.matmul(self.weight)
        if self.use_bias:
            return xw + self.bias
        else:
            return xw

    def extra_repr(self):
        return 'input_dims={}, output_dims={}, extra_dims={}, bias={}, init_type="{}"'.format(
            self.in_features, self.out_features, str(self.extra_dims), self.use_bias, self.init_type
        )
