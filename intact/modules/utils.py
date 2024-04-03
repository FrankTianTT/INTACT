from typing import List, Optional, Union

from torch import nn

from intact.modules.models.layers import ParallelLinear


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
    elif name == "GELU":
        return nn.GELU()
    elif name == "SiLU":
        return nn.SiLU()
    else:
        raise NotImplementedError("{} is not supported as an activate function".format(name))


def check_dims(dims: Union[int, List[int]], name: str = "dims"):
    if not isinstance(dims, list):
        if dims is None:
            dims = []
        elif isinstance(dims, int):
            dims = [dims]
        else:
            raise NotImplementedError("{} should be None or int or list[int]".format(name))
    return dims


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Union[int, Optional[List[int]]] = None,
    extra_dims: Union[int, Optional[List[int]]] = None,
    bias: bool = True,
    drop_out: Optional[float] = None,
    activate_name: str = "ReLU",
    last_activate_name: Optional[str] = None,
    batch_norm: bool = False,
) -> nn.Module:
    hidden_dims = check_dims(hidden_dims, "hidden_dims")
    extra_dims = check_dims(extra_dims, "extra_dims")

    hidden_dims = hidden_dims or []
    all_dims = [input_dim] + hidden_dims + [output_dim]

    layers = []
    for i in range(len(all_dims) - 1):
        if extra_dims is None:
            layers += [nn.Linear(in_features=all_dims[i], out_features=all_dims[i + 1], bias=bias)]
        else:
            layers += [
                ParallelLinear(
                    in_features=all_dims[i],
                    out_features=all_dims[i + 1],
                    extra_dims=extra_dims,
                    bias=bias,
                )
            ]
        if batch_norm:
            layers += [nn.BatchNorm1d(all_dims[i + 1])]
        if drop_out is not None:
            layers += [nn.Dropout(drop_out)]

        if i < len(all_dims) - 2:
            layers += [get_activate(activate_name)]
        else:
            if last_activate_name is not None:
                layers += [get_activate(last_activate_name)]

    return nn.Sequential(*layers)
