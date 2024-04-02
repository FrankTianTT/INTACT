import torch

from intact.modules.utils import build_mlp, get_activate


def test_build_mlp_parallel():
    input_dim = 10
    output_dim = 5
    hidden_dims = [32, 32]
    extra_dims = [2, 3]
    bias = True
    activate_name = "ReLU"
    last_activate_name = "Sigmoid"

    mlp = build_mlp(input_dim, output_dim, hidden_dims, extra_dims, bias, None, activate_name, last_activate_name)
    print(mlp)


def test_build_mlp_plain():
    input_dim = 10
    output_dim = 5
    hidden_dims = [32, 32]
    extra_dims = None
    bias = True
    activate_name = "ReLU"
    last_activate_name = "Sigmoid"

    mlp = build_mlp(input_dim, output_dim, hidden_dims, extra_dims, bias, None, activate_name, last_activate_name)
    print(mlp)


def test_parallel_data():
    from intact.modules.models.causal_mask import CausalMask

    input_dim = 10
    output_dim = 5
    hidden_dims = [32, 32]
    batch_size = 256
    bias = True
    activate_name = "ReLU"
    last_activate_name = "Sigmoid"

    plain_mlp = build_mlp(input_dim, output_dim, hidden_dims, None, bias, None, activate_name, last_activate_name)
    parallel_mlp = build_mlp(input_dim, 1, hidden_dims, output_dim, bias, None, activate_name, last_activate_name)
    causal_mask = CausalMask(input_dim, output_dim, context_input_dim=0, observed_logits_init_bias=0.0)
    inputs = torch.randn(batch_size, input_dim)

    print(causal_mask.printing_mask)
    masked_inputs, mask = causal_mask(inputs)
    print("masked_inputs", masked_inputs.shape)
    o1 = plain_mlp(masked_inputs)
    o2 = parallel_mlp(masked_inputs)
    print(o1.shape)
    print(o2.shape)


def test_get_activate():
    activate_name = "ReLU"
    activate = get_activate(activate_name)
    print(activate)

    activate_name = "ELU"
    activate = get_activate(activate_name)
    print(activate)

    activate_name = "Sigmoid"
    activate = get_activate(activate_name)
    print(activate)

    activate_name = "Tanh"
    activate = get_activate(activate_name)
    print(activate)

    activate_name = "LeakyReLU"
    activate = get_activate(activate_name)
    print(activate)

    activate_name = "Softplus"
    activate = get_activate(activate_name)
    print(activate)

    activate_name = "GELU"
    activate = get_activate(activate_name)
    print(activate)

    activate_name = "SiLU"
    activate = get_activate(activate_name)
    print(activate)

    activate_name = "NotImplemented"
    try:
        activate = get_activate(activate_name)
    except NotImplementedError as e:
        print(e)
