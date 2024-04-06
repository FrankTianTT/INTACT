import math
from itertools import product
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F


# def truncated_normal_(tensor: torch.Tensor, mean: float = 0, std: float = 1) -> torch.Tensor:
#     """Samples from a truncated normal distribution in-place.
#
#     Args:
#         tensor (tensor): the tensor in which sampled values will be stored.
#         mean (float): the desired mean (default = 0).
#         std (float): the desired standard deviation (default = 1).
#
#     Returns:
#         (tensor): the tensor with the stored values. Note that this modifies the input tensor
#             in place, so this is just a pointer to the same object.
#     """
#     torch.nn.init.normal_(tensor, mean=mean, std=std)
#     while True:
#         cond = torch.logical_or(tensor < mean - 2 * std, tensor > mean + 2 * std)
#         bound_violations = torch.sum(cond).item()
#         if bound_violations == 0:
#             break
#         tensor[cond] = torch.normal(mean, std, size=(bound_violations,), device=tensor.device)
#     return tensor


class ParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        extra_dims: Optional[List[int]] = None,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.extra_dims = [] if extra_dims is None else extra_dims

        self.weight = nn.Parameter(
            torch.empty(
                (*extra_dims, out_features, in_features), **factory_kwargs
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((*extra_dims, out_features), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        for dims in product(*map(range, self.extra_dims)):
            nn.init.kaiming_uniform_(self.weight[dims], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.weight[dims]
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[dims], -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ret = x.matmul(self.weight.transpose(-1, -2))
        if self.bias is not None:
            ret += self.bias.unsqueeze(-2)

        return ret

    def extra_repr(self):
        return (
            "in_features={}, out_features={}, extra_dims={}, bias={}".format(
                self.in_features,
                self.out_features,
                self.extra_dims,
                self.bias is not None,
            )
        )


class ParallelGRUCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        extra_dims: Optional[List[int]] = None,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.extra_dims = [] if extra_dims is None else extra_dims
        self.weight_ih = nn.Parameter(
            torch.empty(
                (*self.extra_dims, 3 * self.hidden_size, self.input_size),
                **factory_kwargs,
            )
        )
        self.weight_hh = nn.Parameter(
            torch.empty(
                (*self.extra_dims, 3 * self.hidden_size, self.hidden_size),
                **factory_kwargs,
            )
        )

        if bias:
            self.bias_ih = nn.Parameter(
                torch.empty(
                    (*self.extra_dims, 3 * self.hidden_size), **factory_kwargs
                )
            )
            self.bias_hh = nn.Parameter(
                torch.empty(
                    (*self.extra_dims, 3 * self.hidden_size), **factory_kwargs
                )
            )
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stddev = (
            1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        )
        for weight in self.parameters():
            for dims in product(*map(range, self.extra_dims)):
                nn.init.uniform_(weight[dims], -stddev, stddev)

    def forward(
        self, input: torch.Tensor, hx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """

        :param input: with shape: (*extra_dims, batch_size, input_size) or (*extra_dims, input_size)
        :param hx: with shape: (*extra_dims, batch_size, hidden_size) or (*extra_dims, hidden_size)
        :return:
            output: with same shape as hx
        """
        accepted_dims = (1 + len(self.extra_dims), 2 + len(self.extra_dims))
        if input.dim() not in accepted_dims:
            raise ValueError(
                f"GRUCell: Expected input to be {accepted_dims[0]}D or {accepted_dims[1]}D, "
                f"got {input.dim()}D instead"
            )
        if hx is not None and hx.dim() not in accepted_dims:
            raise ValueError(
                f"GRUCell: Expected hidden to be {accepted_dims[0]}D or {accepted_dims[1]}D, "
                f"got {hx.dim()}D instead"
            )
        is_batched = input.dim() == 2 + len(self.extra_dims)
        if not is_batched:
            input = input.unsqueeze(-2)

        if hx is None:
            hx = torch.zeros(
                *input.shape[:-1],
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
        else:
            hx = hx.unsqueeze(-2) if not is_batched else hx

        ret = parallel_gru_cell(
            input,
            hx,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
        )

        if not is_batched:
            ret = ret.squeeze(0)

        return ret


def parallel_gru_cell(input, hx, weight_ih, weight_hh, bias_ih, bias_hh):
    """
    Perform a GRU cell operation with extra dimensions.

    This function performs a GRU cell operation on the input tensor with the given weights and biases.
    The operation is performed in parallel across the extra dimensions.

    Args:
        input (torch.Tensor): The input tensor of shape (*extra_dims, batch_size, input_size).
        hx (torch.Tensor): The hidden state tensor of shape (*extra_dims, batch_size, hidden_size).
        weight_ih (torch.Tensor): The input-hidden weights tensor of shape (*extra_dims, 3 * hidden_size, input_size).
        weight_hh (torch.Tensor): The hidden-hidden weights tensor of shape (*extra_dims, 3 * hidden_size, hidden_size).
        bias_ih (torch.Tensor): The input-hidden bias tensor of shape (*extra_dims, 3 * hidden_size) or None.
        bias_hh (torch.Tensor): The hidden-hidden bias tensor of shape (*extra_dims, 3 * hidden_size) or None.

    Returns:
        torch.Tensor: The output tensor of shape (*extra_dims, batch_size, hidden_size).
    """

    gi = input.matmul(weight_ih.transpose(-1, -2))
    gh = hx.matmul(weight_hh.transpose(-1, -2))

    if bias_ih is not None:
        gi = gi + bias_ih.unsqueeze(-2)
    if bias_hh is not None:
        gh = gh + bias_hh.unsqueeze(-2)

    i_r, i_u, i_n = gi.chunk(3, dim=-1)
    h_r, h_u, h_n = gh.chunk(3, dim=-1)

    reset_gate = F.sigmoid(i_r + h_r)
    update_gate = F.sigmoid(i_u + h_u)
    new_gate = F.tanh(i_n + reset_gate * h_n)

    hy = (1 - update_gate) * hx + update_gate * new_gate

    return hy
