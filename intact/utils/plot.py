import math
import os

import numpy as np
import torch
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt


def plot_context(
    cfg, world_model, oracle_context, logger=None, log_idx=0, log_prefix="model", color_values=None
):
    context_model = world_model.context_model
    context_gt = torch.stack([v for v in oracle_context.values()], dim=-1).cpu()

    if cfg.model_type == "causal":
        valid_context_idx = world_model.causal_mask.valid_context_idx
    else:
        valid_context_idx = torch.arange(context_model.max_context_dim)

    mcc, permutation, context_hat = context_model.get_mcc(context_gt, valid_context_idx)
    idxes_hat, idxes_gt = permutation

    if color_values is None:
        cmap = None
    else:
        norm = mcolors.Normalize(vmin=min(color_values), vmax=max(color_values))
        cmap = cm.ScalarMappable(norm, plt.get_cmap("Blues")).cmap

    if len(idxes_gt) == 0:
        pass
    elif len(idxes_gt) == 1:
        plt.scatter(
            context_gt[:, idxes_gt[0]], context_hat[:, idxes_hat[0]], c=color_values, cmap=cmap
        )
    else:
        num_rows = math.ceil(math.sqrt(len(idxes_gt)))
        num_cols = math.ceil(len(idxes_gt) / num_rows)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

        context_names = list(oracle_context.keys())
        scatters = []
        for j, (idx_gt, idx_hat) in enumerate(zip(idxes_gt, idxes_hat)):
            ax = axs.flatten()[j]
            ax.set(
                xlabel=context_names[idx_gt],
                ylabel="{}th context".format(valid_context_idx[idx_hat]),
            )
            scatter = ax.scatter(
                context_gt[:, idx_gt], context_hat[:, idx_hat], c=color_values, cmap=cmap
            )
            scatters.append(scatter)

        for j in range(len(idxes_gt), len(axs.flat)):
            axs.flat[j].set_visible(False)

        if color_values is not None and len(scatters) > 0:
            plt.colorbar(scatters[0], ax=axs)

    os.makedirs(log_prefix, exist_ok=True)
    plt.savefig(os.path.join(log_prefix, f"{log_idx}.png"))
    plt.close()
    np.save(os.path.join(log_prefix, f"{log_idx}.npy"), context_hat)

    if logger is not None:
        logger.add_scaler("{}/valid_context_num".format(log_prefix), float(len(valid_context_idx)))
        logger.add_scaler("{}/mcc".format(log_prefix), mcc)
