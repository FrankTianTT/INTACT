from typing import List, Optional

from torch import nn

from tdfa.modules.models.layers import ParallelLinear


def get_activate(name: str = "ReLU"):
    if name == "ReLU":
        return nn.ReLU()
    elif name == "ELU":
        return nn.ELU()
    elif name == "Sigmoid":
        return nn.Sigmoid()
    elif name == "Tanh":
        return nn.Tanh()
    elif name == "LeakyReLU":
        return nn.LeakyReLU()
    elif name == "Softplus":
        return nn.Softplus()
    else:
        raise NotImplementedError("{} is not supported as an activate function".format(name))


def build_parallel_layers(
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        extra_dims: Optional[List[int]] = None,
        bias: bool = True,
        activate_name: str = "ReLu",
        last_activate_name: Optional[str] = None,
) -> nn.Module:
    layers = []
    hidden_dims = [input_dim] + hidden_dims
    for i in range(len(hidden_dims) - 1):
        layers += [ParallelLinear(in_features=hidden_dims[i], out_features=hidden_dims[i + 1],
                                  extra_dims=extra_dims, bias=bias)]
        layers += [get_activate(activate_name)]
    layers += [ParallelLinear(in_features=hidden_dims[-1], out_features=output_dim, extra_dims=extra_dims, bias=bias)]
    if last_activate_name is not None:
        layers += [get_activate(last_activate_name)]
    return nn.Sequential(*layers)
