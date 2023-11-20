import torch
from torch import nn
from torch.distributions import Bernoulli


class CausalMask(nn.Module):
    def __init__(
            self,
            observed_input_dim,
            context_input_dim,
            mask_output_dim,
            logits_clip=3.0,
    ):
        super().__init__()

        self.observed_input_dim = observed_input_dim
        self.context_input_dim = context_input_dim
        self.mask_output_dim = mask_output_dim
        self.logits_clip = logits_clip

        self._mask_logits = nn.Parameter(torch.ones(self.mask_output_dim, self.mask_input_dim) * 10)

    @property
    def mask_input_dim(self):
        return self.observed_input_dim + self.context_input_dim

    @property
    def mask(self):
        return torch.gt(self.mask_logits, 0).int()

    @property
    def mask_logits(self):
        return torch.clamp(self._mask_logits, -self.logits_clip, self.logits_clip)

    def forward(self, inputs, dim_map=None, deterministic=False):
        assert len(inputs.shape) == 2, "inputs should be 2D tensor: batch_size x input_dim"
        batch_size, input_dim = inputs.shape
        if dim_map is None:
            assert input_dim == self.mask_input_dim, \
                "dimension of inputs should be equal to mask_input_dim if dim_map is None"

        # shape: mask_output_dim * batch_size * input_dim
        repeated_inputs = inputs.unsqueeze(0).expand(self.mask_output_dim, -1, -1)

        if deterministic:
            original_mask = torch.gt(self.mask_logits, 0).float().expand(batch_size, -1, -1)
        else:
            original_mask = Bernoulli(logits=self.mask_logits).sample(torch.Size([batch_size]))

        if dim_map is not None:
            mask = original_mask[:, :, dim_map]
        else:
            mask = original_mask

        masked_inputs = torch.einsum("boi,obi->obi", mask, repeated_inputs)

        return masked_inputs, mask


if __name__ == '__main__':
    observed_input_dim = 5
    context_input_dim = 10
    real_input_dim = 100
    mask_output_dim = 20
    batch_size = 5

    mask = CausalMask(
        observed_input_dim=observed_input_dim,
        context_input_dim=context_input_dim,
        mask_output_dim=mask_output_dim,
    )

    inputs = torch.randn(batch_size, real_input_dim)
    dim_map = torch.randint(0, observed_input_dim + context_input_dim, (real_input_dim,))
    print(dim_map)

    masked_inputs, mask = mask(inputs, dim_map=dim_map)

    print(masked_inputs.shape)
    print(mask.shape)
