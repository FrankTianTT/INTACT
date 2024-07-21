import pdb

import torch
from torch import nn
from torch.distributions import Bernoulli
from torch.nn import functional as F


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
            mask_type="direct",
            sigmoid_threshold=0.1,
            latent=False,
            gumbel_softmax=False,
            meta=False,
            context_input_dim=10,
            logits_clip=3.0,
            observed_logits_init_bias=0.0,
            context_logits_init_bias=0.0,
            observed_logits_init_scale=0.0,
            context_logits_init_scale=0.0,
            alpha=10.0,
    ):
        """
        Initialize the CausalMask.

        Args:
            observed_input_dim (int): The number of observed input dimensions.
            mask_output_dim (int): The number of mask output dimensions.
            latent (bool, optional): Whether to use latent variables. Defaults to False.
            gumbel_softmax (bool, optional): Whether to use Gumbel-Softmax for relaxation. Defaults to False.
            meta (bool, optional): Whether to use meta-learning. Defaults to False.
            context_input_dim (int, optional): The number of context input dimensions. Defaults to 10.
            logits_clip (float, optional): The value to clip the logits at. Defaults to 3.0.
            observed_logits_init_bias (float, optional): The initial bias for the observed logits. Defaults to 0.3.
            context_logits_init_bias (float, optional): The initial bias for the context logits. Defaults to 0.5.
            observed_logits_init_scale (float, optional): The initial scale for the observed logits. Defaults to 0.05.
            context_logits_init_scale (float, optional): The initial scale for the context logits. Defaults to 0.5.
            alpha (float, optional): The temperature parameter for the sigmoid function. Defaults to 10.0.
        """
        super().__init__()

        self.observed_input_dim = observed_input_dim
        self.mask_output_dim = mask_output_dim
        self.mask_type = mask_type
        self.sigmoid_threshold = sigmoid_threshold
        self.latent = latent
        self.gumbel_softmax = gumbel_softmax
        self.meta = meta
        self.context_input_dim = context_input_dim if meta else 0
        self.logits_clip = logits_clip
        self.observed_logits_init_bias = observed_logits_init_bias
        self.context_logits_init_bias = context_logits_init_bias
        self.observed_logits_init_scale = observed_logits_init_scale
        self.context_logits_init_scale = context_logits_init_scale
        self.alpha = alpha

        init_observed = (torch.randn(self.mask_output_dim, self.observed_input_dim) * observed_logits_init_scale
                         + observed_logits_init_bias)
        init_context = (torch.randn(self.mask_output_dim, self.context_input_dim) * context_logits_init_scale
                        + context_logits_init_bias)

        if self.mask_type == "direct":
            self._observed_logits = nn.Parameter(init_observed + 0.5)
            self._context_logits = nn.Parameter(init_context + 0.5)
        else:
            self._observed_logits = nn.Parameter(init_observed)
            self._context_logits = nn.Parameter(init_context)

    def extra_repr(self):
        """
        Return a string representation of the CausalMask.

        Returns:
            str: A string representation of the CausalMask.
        """
        return "observed_input_dim={}, mask_output_dim={}, context_input_dim={}".format(
            self.observed_input_dim,
            self.mask_output_dim,
            self.context_input_dim,
        )

    @property
    def mask_input_dim(self):
        """
        Get the total mask input dimension.

        Returns:
            int: The total mask input dimension.
        """
        return self.observed_input_dim + self.context_input_dim

    def get_parameter(self, target: str):
        """
        Get the parameters of the specified target.

        Args:
            target (str): The target to get the parameters of.

        Returns:
            list: A list containing the parameters of the target.

        Raises:
            NotImplementedError: If the target is not recognized.
        """
        if target == "observed_logits":
            return [self._observed_logits]
        elif target == "context_logits":
            return [self._context_logits]
        else:
            raise NotImplementedError

    @property
    def mask(self):
        if self.mask_type == "reinforce":
            return torch.gt(self.mask_logits, 0).int()
        elif self.mask_type == "sigmoid" or self.mask_type == "gumbel":
            return torch.gt(torch.sigmoid(self.alpha * self.mask_logits), self.sigmoid_threshold).int()
        elif self.mask_type == "direct":
            return torch.gt(self.mask_logits, 0).int()
        else:
            raise NotImplemented
        
    @property
    def soft_mask(self):
        if self.mask_type == "direct":
            return self.mask_logits
        else:
            return torch.sigmoid(self.alpha * self.mask_logits)

    @property
    def mask_logits(self):
        return torch.cat([self.observed_logits, self.context_logits], dim=-1)

    @property
    def observed_logits(self):
        """
        Get the observed logits.

        Returns:
            Tensor: The observed logits.
        """
        if self.mask_type == "direct":
            return torch.clamp(self._observed_logits, 0, 1)
        else:
            return torch.clamp(self._observed_logits, -self.logits_clip, self.logits_clip)

    @property
    def context_logits(self):
        """
        Get the context logits.

        Returns:
           Tensor: The context logits.
        """
        if self.mask_type == "direct":
            return torch.clamp(self._context_logits, 0, 1)
        else:
            return torch.clamp(self._observed_logits, -self.logits_clip, self.logits_clip)

    @property
    def valid_context_idx(self):
        """
        Get the valid context indices.

        Returns:
            Tensor: The valid context indices.
        """
        non_zero = self.mask[:, self.observed_input_dim:].any(dim=0)
        return torch.where(non_zero)[0]

    def reset(self, line_idx=None):
        """
        Reset the context logits.

        Args:
          line_idx (list, optional): The indices of the lines to reset. Defaults to None.
        """
        if line_idx is None:
            line_idx = list(range(self.mask_output_dim))

        column_idx = list(set(range(self.context_input_dim)) - set(self.valid_context_idx.tolist()))
        line_matrix = torch.zeros_like(self._context_logits).to(bool)
        column_matrix = torch.zeros_like(self._context_logits).to(bool)
        line_matrix[line_idx] = 1
        column_matrix[:, column_idx] = 1
        reset_matrix = line_matrix * column_matrix

        if self.model_type == "direct":
            self._context_logits.data[reset_matrix] = 0.5
        else:
            self._context_logits.data[reset_matrix] = get_init(
                shape=(len(line_idx) * len(column_idx)),
                bias=self.context_logits_init_bias,
                scale=self.context_logits_init_scale,
            ).to(self._context_logits.device)

    def forward(self, inputs, dim_map=None, deterministic=False):
        assert len(inputs.shape) == 2, "inputs should be 2D tensor: batch_size x input_dim"
        batch_size, input_dim = inputs.shape
        if dim_map is None:
            assert input_dim == self.mask_input_dim, (
                f"dimension of inputs should be equal to mask_input_dim if dim_map"
                f" is None, got {input_dim} and {self.mask_input_dim} instead"
            )

        # shape: mask_output_dim * batch_size * input_dim
        repeated_inputs = inputs.unsqueeze(0).expand(self.mask_output_dim, -1, -1)

        if self.mask_type == "reinforce":
            if deterministic:
                original_mask = self.mask.float().expand(batch_size, -1, -1)
            else:
                original_mask = Bernoulli(logits=self.mask_logits).sample(torch.Size([batch_size]))
        else:
            if self.mask_type == "gumbel":
                original_mask = F.gumbel_softmax(
                    logits=torch.stack((self.mask_logits, 1 - self.mask_logits)),
                    hard=True,
                    dim=0,
                )[0]
            elif self.mask_type == "sigmoid":
                original_mask = torch.sigmoid(self.alpha * self.mask_logits)
            elif self.mask_type == "direct":
                original_mask = self.mask_logits
            else:
                raise NotImplemented
            original_mask = original_mask.expand(batch_size, -1, -1)

        mask = original_mask[:, :, dim_map] if dim_map is not None else original_mask
        masked_inputs = torch.einsum("boi,obi->obi", mask, repeated_inputs)
        return masked_inputs, original_mask

    def total_mask_grad(
            self,
            sampling_mask,
            sampling_loss,
            sparse_weight=0.05,
            context_sparse_weight=0.05,
            context_max_weight=0.2,
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
        reg_grad[:, : self.observed_input_dim] *= sparse_weight
        # if self.latent:
        #     max_idx = self.mask_logits[:, :self.observed_input_dim].argmax(dim=1)
        #     reg_grad[torch.arange(self.mask_output_dim), max_idx] = 0
        reg_grad[:, self.observed_input_dim:] *= context_sparse_weight
        reg_grad[:, self.observed_input_dim:] += context_max_weight * max_sigmoid_grad(
            self.mask_logits[:, self.observed_input_dim:]
        )
        grad = is_valid * (sampling_grad + reg_grad)
        return grad.mean(dim=0)

    @property
    def printing_mask(self):
        mask = self.mask
        print(self.mask_logits)
        string = ""
        for i, out_dim in enumerate(range(mask.shape[0])):
            string += " ".join([str(ele.item()) for ele in mask[out_dim]])
            if i != mask.shape[0] - 1:
                string += "\n"
        return string
