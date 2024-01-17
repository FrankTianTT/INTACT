import math
import os

import torch
from torch.nn.utils import clip_grad_norm_
import tensordict

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import cm

from itertools import product
from functools import partial
from copy import deepcopy
import os
import math
from warnings import catch_warnings, filterwarnings

from tqdm import tqdm
import torch
from torchrl.envs import SerialEnv
from torchrl.collectors.collectors import aSyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import cm

from causal_meta.objectives.causal_dreamer import CausalDreamerModelLoss
from causal_meta.utils.eval import evaluate_policy
from causal_meta.utils.envs import make_dreamer_env


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


def train_model(
        cfg,
        replay_buffer,
        world_model,
        world_model_loss,
        model_opt,
        training_steps,
        logits_opt=None,
        logger=None,
        deterministic_mask=False,
        log_prefix="model",
        iters=0,
        only_train=None
):
    device = next(world_model.parameters()).device
    train_logits_by_reinforce = cfg.model_type == "causal" and cfg.reinforce
    if train_logits_by_reinforce:
        assert logits_opt is not None, "logits_opt should not be None when train logits by reinforce"

    if cfg.model_type == "causal":
        causal_mask = world_model.causal_mask
    else:
        causal_mask = None

    for step in range(training_steps):
        world_model.zero_grad()
        sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(device, non_blocking=True)

        if (train_logits_by_reinforce and
                iters % (cfg.train_mask_iters + cfg.train_model_iters) >= cfg.train_model_iters):
            grad, sampling_loss = world_model_loss.reinforce(sampled_tensordict)
            causal_mask = world_model.causal_mask
            logits = causal_mask.mask_logits
            logits.backward(grad)
            logits_opt.step()
        else:
            model_loss_td, sampled_tensordict = world_model_loss(sampled_tensordict)
            total_loss = sum([loss for loss in model_loss_td.values()])

            total_loss.backward()
            clip_grad_norm_(world_model.get_parameter("nets"), cfg.grad_clip)
            model_opt.step()

            logger.add_scaler("world_model/total_loss", total_loss)
            logger.add_scaler("world_model/kl_loss", model_loss_td["loss_model_kl"])
            logger.add_scaler("world_model/reco_loss", model_loss_td["loss_model_reco"])
            logger.add_scaler("world_model/reward_loss", model_loss_td["loss_model_reward"])
            logger.add_scaler("world_model/continue_loss", model_loss_td["loss_model_continue"])

        if cfg.model_type == "causal":
            mask_value = torch.sigmoid(cfg.alpha * causal_mask.mask_logits)
            for out_dim, in_dim in product(range(mask_value.shape[0]), range(mask_value.shape[1])):
                out_name = f"o{out_dim}"
                if in_dim < causal_mask.observed_input_dim:
                    in_name = f"i{in_dim}"
                else:
                    in_name = f"c{in_dim - causal_mask.observed_input_dim}"
                logger.add_scaler(f"{log_prefix}/mask_value({out_name},{in_name})", mask_value[out_dim, in_dim])

        iters += 1
    return iters


def train_agent(
        cfg,
        replay_buffer,
        actor_model,
        actor_loss,
        actor_opt,
        value_model,
        value_loss,
        value_opt,
        training_steps,
        logger=None,
):
    device = next(actor_model.parameters()).device

    for step in range(training_steps):
        sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(device, non_blocking=True)

        actor_loss_td, sampled_tensordict = actor_loss(sampled_tensordict)
        actor_loss_td["loss_actor"].backward()
        clip_grad_norm_(actor_model.parameters(), cfg.grad_clip)
        actor_opt.step()

        logger.add_scaler("policy/loss", actor_loss_td["loss_actor"])
        logger.add_scaler("policy/grad", grad_norm(actor_opt))
        logger.add_scaler("policy/action_mean", sampled_tensordict["action"].mean())
        logger.add_scaler("policy/action_std", sampled_tensordict["action"].std())
        actor_opt.zero_grad()

        # update value network
        value_loss_td, sampled_tensordict = value_loss(sampled_tensordict)
        value_loss_td["loss_value"].backward()
        clip_grad_norm_(value_model.parameters(), cfg.grad_clip)
        value_opt.step()

        logger.add_scaler("value/loss", value_loss_td["loss_value"])
        logger.add_scaler("value/grad", grad_norm(value_opt))
        logger.add_scaler("value/target_mean", sampled_tensordict["lambda_target"].mean())
        logger.add_scaler("value/target_std", sampled_tensordict["lambda_target"].std())
        logger.add_scaler("value/mean_continue",
                          (sampled_tensordict[("next", "pred_continue")] > 0).float().mean())
        value_opt.zero_grad()


def reset_module(policy, world_model, new_domain_task_num):
    device = next(policy.parameters()).device
    new_policy = deepcopy(policy).to(device)
    new_world_model = deepcopy(world_model).to(device)

    new_context_model = new_world_model.context_model
    new_context_model.reset(new_domain_task_num)
    new_actor = new_policy[2][0]
    new_actor.set_context_model(new_context_model)

    return new_policy, new_world_model


def meta_test(
        cfg,
        make_env_list,
        oracle_context,
        policy,
        world_model,
        logger,
        frames_per_task,
        adapt_threshold=-3.
):
    logger.dump_scaler(frames_per_task)

    task_num = len(make_env_list)

    policy, world_model = reset_module(policy, world_model, task_num)
    device = next(world_model.parameters()).device

    world_model_loss = CausalDreamerModelLoss(
        world_model,
        lambda_kl=cfg.lambda_kl,
        lambda_reco=cfg.lambda_reco,
        lambda_reward=cfg.lambda_reward,
        lambda_continue=cfg.lambda_continue,
        free_nats=cfg.free_nats,
        sparse_weight=cfg.sparse_weight,
        context_sparse_weight=cfg.context_sparse_weight,
        context_max_weight=cfg.context_max_weight,
        sampling_times=cfg.sampling_times,
    ).to(device)

    collector = aSyncDataCollector(
        create_env_fn=SerialEnv(task_num, make_env_list, shared_memory=False),
        policy=policy,
        total_frames=cfg.meta_task_adjust_frames_per_task,
        frames_per_batch=500,
        init_random_frames=0,
        device=cfg.collector_device,
        storing_device=cfg.collector_device,
        split_trajs=True
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(max_size=cfg.buffer_size),
    )
    world_model_opt = torch.optim.Adam(world_model.get_parameter("context"), lr=cfg.context_lr)

    pbar = tqdm(total=cfg.meta_task_adjust_frames_per_task, desc="meta_test_adjust")
    train_model_iters = 0
    for frame, tensordict in enumerate(collector):
        current_frames = tensordict.get(("collector", "mask")).sum().item()
        pbar.update(current_frames)
        tensordict = match_length(tensordict, cfg.batch_length)
        replay_buffer.extend(tensordict.reshape(-1))

        train_model_iters = train_model(
            cfg, replay_buffer, world_model, world_model_loss, world_model_opt,
            cfg.optim_steps_per_batch, None, logger,
            iters=train_model_iters, log_prefix=f"meta_test_model_{frames_per_task}",
        )

        plot_context(cfg, world_model, oracle_context, logger, frame, log_prefix=f"meta_test_model_{frames_per_task}")
        logger.dump_scaler(frame)

    evaluate_policy(
        cfg, oracle_context, policy, logger, frames_per_task, log_prefix="meta_test",
        make_env_fn=partial(
            make_dreamer_env,
            variable_num=cfg.variable_num,
            state_dim_per_variable=cfg.state_dim_per_variable,
            hidden_dim_per_variable=cfg.belief_dim_per_variable
        ),
        disable_pixel_if_possible=False
    )
