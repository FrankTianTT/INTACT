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


def get_init(shape, bias, scale):
    return torch.randn(shape) * scale + bias


class CausalMask(nn.Module):
    def __init__(
        self,
        observed_input_dim,
        mask_output_dim,
        using_reinforce=True,
        latent=False,
        gumbel_softmax=False,
        meta=False,
        context_input_dim=10,
        logits_clip=3.0,
        observed_logits_init_bias=0.3,
        context_logits_init_bias=0.5,
        observed_logits_init_scale=0.05,
        context_logits_init_scale=0.5,
        alpha=10.0,
    ):
        """
        Initialize the CausalMask.

        Args:
            observed_input_dim (int): The number of observed input dimensions.
            mask_output_dim (int): The number of mask output dimensions.
            using_reinforce (bool, optional): Whether to use REINFORCE for gradient estimation. Defaults to True.
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
        self.using_reinforce = using_reinforce
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

        self._observed_logits = nn.Parameter(
            get_init(
                shape=(self.mask_output_dim, self.observed_input_dim),
                bias=observed_logits_init_bias,
                scale=observed_logits_init_scale,
            )
        )
        # self._observed_logits = nn.Parameter(torch.ones((self.mask_output_dim, self.observed_input_dim)) * logits_clip)

        self._context_logits = nn.Parameter(
            get_init(
                shape=(self.mask_output_dim, self.context_input_dim),
                bias=context_logits_init_bias,
                scale=context_logits_init_scale,
            )
        )

    def extra_repr(self):
        """
        Return a string representation of the CausalMask.

        Returns:
            str: A string representation of the CausalMask.
        """
        return "observed_input_dim={}, mask_output_dim={}, context_input_dim={}".format(
            self.observed_input_dim, self.mask_output_dim, self.context_input_dim
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
        return torch.gt(self.mask_logits, 0).int()

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
        return torch.clamp(self._observed_logits, -self.logits_clip, self.logits_clip)

    @property
    def context_logits(self):
        """
        Get the context logits.

        Returns:
           Tensor: The context logits.
        """
        return torch.clamp(self._context_logits, -self.logits_clip, self.logits_clip)

    @property
    def valid_context_idx(self):
        """
        Get the valid context indices.

        Returns:
            Tensor: The valid context indices.
        """
        non_zero = self.mask[:, self.observed_input_dim :].any(dim=0)
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
                "dimension of inputs should be equal to mask_input_dim if dim_map is None,"
                "got {} and {} instead".format(input_dim, self.mask_input_dim)
            )

        # shape: mask_output_dim * batch_size * input_dim
        repeated_inputs = inputs.unsqueeze(0).expand(self.mask_output_dim, -1, -1)

        if self.using_reinforce:
            if deterministic:
                original_mask = self.mask.float().expand(batch_size, -1, -1)
            else:
                original_mask = Bernoulli(logits=self.mask_logits).sample(torch.Size([batch_size]))
        else:
            if self.gumbel_softmax:
                original_mask = F.gumbel_softmax(
                    logits=torch.stack((self.mask_logits, 1 - self.mask_logits)), hard=True, dim=0
                )[0]
            else:
                original_mask = torch.sigmoid(self.alpha * self.mask_logits)
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
        pos_grads = torch.einsum("sbo,sboi->sboi", sampling_loss, sampling_mask).sum(dim=0) / (
            num_pos + 1e-6
        )
        neg_grads = torch.einsum("sbo,sboi->sboi", sampling_loss, 1 - sampling_mask).sum(dim=0) / (
            num_neg + 1e-6
        )

        g = self.mask_logits.sigmoid() * (1 - self.mask_logits.sigmoid())

        sampling_grad = (pos_grads - neg_grads) * g
        reg_grad = torch.ones_like(self.mask_logits)
        reg_grad[:, : self.observed_input_dim] *= sparse_weight
        # if self.latent:
        #     max_idx = self.mask_logits[:, :self.observed_input_dim].argmax(dim=1)
        #     reg_grad[torch.arange(self.mask_output_dim), max_idx] = 0
        reg_grad[:, self.observed_input_dim :] *= context_sparse_weight
        reg_grad[:, self.observed_input_dim :] += context_max_weight * max_sigmoid_grad(
            self.mask_logits[:, self.observed_input_dim :]
        )
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
