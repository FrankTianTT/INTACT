import math
import os

import torch
import tensordict

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import cm

def grad_norm(optimizer: torch.optim.Optimizer):
    sum_of_sq = 0.0
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            if p.grad is not None:
                sum_of_sq += p.grad.pow(2).sum()
    return sum_of_sq.sqrt().detach().item()


def retrieve_stats_from_state_dict(obs_norm_state_dict):
    return {
        "loc": obs_norm_state_dict["loc"],
        "scale": obs_norm_state_dict["scale"],
    }


def match_length(batch_td: tensordict.TensorDict, length):
    assert len(batch_td.shape) == 2, "batch_td must be 2D"

    batch_size, seq_len = batch_td.shape
    # min multiple of length that larger than or equal to seq_len
    new_seq_len = (seq_len + length - 1) // length * length

    # pad with zeros
    matched_td = torch.stack(
        [tensordict.pad(td, [0, new_seq_len - seq_len]) for td in batch_td], 0
    ).contiguous()
    return matched_td


def plot_context(
        cfg,
        world_model,
        oracle_context,
        logger=None,
        frames_per_task=0,
        log_prefix="model",
        plot_path="",
        color_values=None
):
    context_model = world_model.context_model
    context_gt = torch.stack([v for v in oracle_context.values()], dim=-1).cpu()

    if cfg.model_type == "causal":
        valid_context_idx = world_model.causal_mask.valid_context_idx
    else:
        valid_context_idx = torch.arange(context_model.max_context_dim)

    mcc, permutation, context_hat = context_model.get_mcc(context_gt, valid_context_idx)
    idxes_hat, idxes_gt = permutation

    os.makedirs(log_prefix, exist_ok=True)

    if color_values is None:
        cmap = None
    else:
        norm = mcolors.Normalize(vmin=min(color_values), vmax=max(color_values))
        cmap = cm.ScalarMappable(norm, plt.get_cmap('Blues')).cmap

    if len(idxes_gt) == 0:
        pass
    elif len(idxes_gt) == 1:
        plt.scatter(context_gt[:, idxes_gt[0]], context_hat[:, idxes_hat[0]], c=color_values, cmap=cmap)
    else:
        num_rows = math.ceil(math.sqrt(len(idxes_gt)))
        num_cols = math.ceil(len(idxes_gt) / num_rows)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

        context_names = list(oracle_context.keys())
        scatters = []
        for j, (idx_gt, idx_hat) in enumerate(zip(idxes_gt, idxes_hat)):
            ax = axs.flatten()[j]
            ax.set(xlabel=context_names[idx_gt], ylabel="{}th context".format(valid_context_idx[idx_hat]))
            scatter = ax.scatter(context_gt[:, idx_gt], context_hat[:, idx_hat], c=color_values, cmap=cmap)
            scatters.append(scatter)

        for j in range(len(idxes_gt), len(axs.flat)):
            axs.flat[j].set_visible(False)

        if color_values is not None and len(scatters) > 0:
            plt.colorbar(scatters[0], ax=axs)

    if plot_path == "":
        plot_path = os.path.join(log_prefix, f"{frames_per_task}.png")
    plt.savefig(plot_path)
    plt.close()

    if logger is not None:
        logger.add_scaler("{}/valid_context_num".format(log_prefix), float(len(valid_context_idx)))
        logger.add_scaler("{}/mcc".format(log_prefix), mcc)
