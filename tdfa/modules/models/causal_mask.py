import torch
from torch import nn
from torch.distributions import Bernoulli


def max_sigmoid_grad(logits):
    """calculate the gradient of max_sigmoid_grad function.

    :param logits: a 2d tensor of logits
    :return:
        grad: gradient of the max_sigmoid_grad function
    """
    assert len(logits.shape) == 2

    max_val, _ = torch.max(logits, dim=0, keepdim=True)

    equal_max = torch.eq(logits, max_val)
    max_val_grad = torch.sigmoid(max_val) * (1 - torch.sigmoid(max_val))

    grad = torch.where(equal_max, max_val_grad, torch.zeros_like(logits))
    return grad


class CausalMask(nn.Module):
    def __init__(
            self,
            observed_input_dim,
            mask_output_dim,
            meta=False,
            context_input_dim=10,
            logits_clip=3.0,
            logits_init_bias=1.0,
            logits_init_scale=0.05,
    ):
        super().__init__()

        self.observed_input_dim = observed_input_dim
        self.mask_output_dim = mask_output_dim
        self.meta = meta
        self.context_input_dim = context_input_dim if meta else 0
        self.logits_clip = logits_clip
        self.logits_init_bias = logits_init_bias
        self.logits_init_scale = logits_init_scale

        init_logits = torch.randn(self.mask_output_dim, self.mask_input_dim) * logits_init_scale + logits_init_bias
        self._mask_logits = nn.Parameter(init_logits)

    @property
    def mask_input_dim(self):
        return self.observed_input_dim + self.context_input_dim

    @property
    def mask(self):
        return torch.gt(self.mask_logits, 0).int()

    @property
    def mask_logits(self):
        return torch.clamp(self._mask_logits, -self.logits_clip, self.logits_clip)

    @property
    def valid_context_idx(self):
        non_zero = self.mask[:, -self.context_input_dim:].any(dim=0)
        return torch.where(non_zero)[0]

    def forward(self, inputs, dim_map=None, deterministic=False):
        assert len(inputs.shape) == 2, "inputs should be 2D tensor: batch_size x input_dim"
        batch_size, input_dim = inputs.shape
        if dim_map is None:
            assert input_dim == self.mask_input_dim, \
                ("dimension of inputs should be equal to mask_input_dim if dim_map is None,"
                 "got {} and {} instead".format(input_dim, self.mask_input_dim))

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

        return masked_inputs, original_mask

    import torch

    def total_mask_grad(
            self,
            sampling_mask,
            sampling_loss,
            sparse_weight=0.05,
            context_sparse_weight=0.05,
            context_max_weight=0.2
    ):
        num_pos = sampling_mask.sum(dim=0)
        num_neg = sampling_mask.shape[0] - num_pos

        # one sample is valid if its ``sampling_mask`` contains both positive and negative logit
        is_valid = ((num_pos > 0) * (num_neg > 0)).float()

        # calculate the gradient of the sampling loss w.r.t. the logits
        pos_grads = torch.einsum("sbo,sboi->sboi", sampling_loss, sampling_mask).sum(dim=0) / (num_pos + 1e-6)
        neg_grads = torch.einsum("sbo,sboi->sboi", sampling_loss, 1 - sampling_mask).sum(dim=0) / (num_neg + 1e-6)

        g = self.mask_logits.sigmoid() * (1 - self.mask_logits.sigmoid())

        sampling_grad = (pos_grads - neg_grads) * g
        reg_grad = torch.ones_like(self.mask_logits)
        reg_grad[:, :self.observed_input_dim] *= sparse_weight
        reg_grad[:, self.observed_input_dim:] *= context_sparse_weight
        reg_grad[:, self.observed_input_dim:] += (context_max_weight *
                                                  max_sigmoid_grad(self.mask_logits[:, self.observed_input_dim:]))

        grad = is_valid * (sampling_grad + reg_grad)
        return grad.mean(dim=0)

    @property
    def printing_mask(self):
        string = ""
        for i, out_dim in enumerate(range(self.mask.shape[0])):
            string += " ".join([str(ele.item()) for ele in self.mask[out_dim]])
            if i != self.mask.shape[0] - 1:
                string += "\n"
        return string


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
