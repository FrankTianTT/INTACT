import torch

from intact.modules.models.causal_mask import CausalMask


def test_causal_mask_reinforce():
    observed_input_dim = 5
    context_input_dim = 10
    real_input_dim = 100
    mask_output_dim = 6
    batch_size = 5

    causal_mask = CausalMask(
        observed_input_dim=observed_input_dim,
        context_input_dim=context_input_dim,
        mask_output_dim=mask_output_dim,
        meta=True,
    )

    inputs = torch.randn(batch_size, real_input_dim)
    dim_map = torch.randint(0, observed_input_dim + context_input_dim, (real_input_dim,))

    masked_inputs, mask = causal_mask(inputs, dim_map=dim_map)

    assert masked_inputs.shape == (mask_output_dim, batch_size, real_input_dim)
    assert mask.shape == (batch_size, mask_output_dim, context_input_dim + observed_input_dim)

    print(causal_mask.mask)
    causal_mask.reset()
    print(causal_mask.get_parameter("context_logits"))


def test_causal_mask_sigmoid():
    observed_input_dim = 5
    context_input_dim = 10
    real_input_dim = 100
    mask_output_dim = 6
    batch_size = 5

    causal_mask = CausalMask(
        observed_input_dim=observed_input_dim,
        context_input_dim=context_input_dim,
        mask_output_dim=mask_output_dim,
        meta=True,
        using_reinforce=False,
    )

    inputs = torch.randn(batch_size, real_input_dim)
    dim_map = torch.randint(0, observed_input_dim + context_input_dim, (real_input_dim,))

    masked_inputs, _ = causal_mask(inputs, dim_map=dim_map)

    assert masked_inputs.shape == (mask_output_dim, batch_size, real_input_dim)
