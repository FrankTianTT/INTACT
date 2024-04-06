import torch

from intact.modules.models.layers import ParallelLinear, ParallelGRUCell


def test_parallel_linear():
    in_features = 3
    out_features = 5
    extra_dims = [10]
    batch_size = 32

    parallel_linear = ParallelLinear(
        in_features=in_features,
        out_features=out_features,
        extra_dims=extra_dims,
    )

    inputs = torch.randn(*extra_dims, batch_size, in_features)
    outputs = parallel_linear(inputs)

    assert outputs.shape == (*extra_dims, batch_size, out_features)


def test_parallel_gru_cell():
    input_size = 3
    hidden_size = 20
    extra_dims = [10]
    batch_size = 32

    parallel_gru_cell = ParallelGRUCell(
        input_size=input_size,
        hidden_size=hidden_size,
        extra_dims=extra_dims,
        bias=True,
    )

    inputs = torch.randn(*extra_dims, batch_size, input_size)
    hx = torch.randn(*extra_dims, batch_size, hidden_size)

    outputs = parallel_gru_cell(inputs, hx)
    assert outputs.shape == (*extra_dims, batch_size, hidden_size)
