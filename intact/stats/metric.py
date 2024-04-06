import torch

from intact.stats.kernel import kernel_classes


def mutual_info_estimation(
    values, bandwidth=1.0, kernel_type="gaussian", reduction="mean"
):
    # values: batch * dim
    kernel_class = kernel_classes[kernel_type]

    kernel = kernel_class(bandwidth=bandwidth)
    joint_log_pdf = kernel(values, values)

    sum_margin_log_pdf = 0
    for i in range(values.shape[1]):
        log_pdf = kernel(values[:, i : i + 1], values[:, i : i + 1])
        sum_margin_log_pdf += log_pdf

    if reduction == "mean":
        return torch.mean(joint_log_pdf - sum_margin_log_pdf)
    elif reduction == "none":
        return joint_log_pdf - sum_margin_log_pdf
    else:
        raise NotImplementedError


def kurtoses_estimation(values):
    # values: batch * dim
    mean = torch.mean(values, dim=0)
    diffs = values - mean
    var = torch.mean(torch.pow(diffs, 2.0), dim=0)
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    kurtoses = torch.mean(torch.pow(zscores, 4.0), dim=0) - 3.0
    return kurtoses
