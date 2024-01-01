from typing import List, Optional

from torch import nn

from causal_meta.modules.models.layers import ParallelLinear


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


def build_mlp(
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        extra_dims: Optional[List[int]] = None,
        bias: bool = True,
        activate_name: str = "ReLU",
        last_activate_name: Optional[str] = None,
) -> nn.Module:
    hidden_dims = hidden_dims or []
    all_dims = [input_dim] + hidden_dims + [output_dim]

    layers = []
    for i in range(len(all_dims) - 1):
        if extra_dims is None:
            layers += [nn.Linear(in_features=all_dims[i], out_features=all_dims[i + 1], bias=bias)]
        else:
            layers += [ParallelLinear(in_features=all_dims[i], out_features=all_dims[i + 1],
                                      extra_dims=extra_dims, bias=bias)]

        if i < len(all_dims) - 2:
            layers += [get_activate(activate_name)]
        else:
            if last_activate_name is not None:
                layers += [get_activate(last_activate_name)]

    return nn.Sequential(*layers)


def test_build_mlp_parallel():
    input_dim = 10
    output_dim = 5
    hidden_dims = [32, 32]
    extra_dims = [2, 3]
    bias = True
    activate_name = "ReLU"
    last_activate_name = "Sigmoid"

    mlp = build_mlp(input_dim, output_dim, hidden_dims, extra_dims, bias, activate_name, last_activate_name)
    print(mlp)


def test_build_mlp_plain():
    input_dim = 10
    output_dim = 5
    hidden_dims = [32, 32]
    extra_dims = None
    bias = True
    activate_name = "ReLU"
    last_activate_name = "Sigmoid"

    mlp = build_mlp(input_dim, output_dim, hidden_dims, extra_dims, bias, activate_name, last_activate_name)
    print(mlp)


if __name__ == '__main__':
    test_build_mlp_parallel()
    test_build_mlp_plain()
