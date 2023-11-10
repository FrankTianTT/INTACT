import torch


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


def total_mask_grad(
        logits,
        sampling_mask,
        sampling_loss,
        observed_input_dim,
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

    g = logits.sigmoid() * (1 - logits.sigmoid())

    sampling_grad = (pos_grads - neg_grads) * g
    reg_grad = torch.ones_like(logits)
    reg_grad[:, :observed_input_dim] *= sparse_weight
    reg_grad[:, observed_input_dim:] *= context_sparse_weight
    reg_grad[:, observed_input_dim:] += (context_max_weight *
                                         max_sigmoid_grad(logits[:, observed_input_dim:]))

    grad = is_valid * (sampling_grad + reg_grad)
    return grad.mean(dim=0)
