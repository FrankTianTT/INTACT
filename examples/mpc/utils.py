from itertools import product
from functools import partial
from copy import deepcopy
import os
import math
from warnings import catch_warnings, filterwarnings

from tqdm import tqdm
import torch
from tensordict.nn import TensorDictModule, TensorDictModuleWrapper
from tensordict.nn.probabilistic import set_interaction_mode
from torchrl.envs.utils import step_mdp
from torchrl.envs import TransformedEnv, SerialEnv
from torchrl.record import VideoRecorder
from torchrl.trainers.helpers.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import ListStorage
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import cm

from causal_meta.helpers.envs import build_make_env_list, make_mdp_env
from causal_meta.objectives.causal_mdp import CausalWorldModelLoss
from causal_meta.envs.mdp_env import MDPEnv


class MultiOptimizer:
    def __init__(self, **optimizers):
        self.optimizers = optimizers

    def step(self):
        for opt in self.optimizers.values():
            opt.step()


def evaluate_policy(
        cfg,
        oracle_context,
        policy,
        logger,
        frames_per_task,
        log_prefix="meta_train",
):
    device = next(policy.parameters()).device

    pbar = tqdm(total=cfg.eval_repeat_nums * cfg.env_max_steps, desc="{}_eval".format(log_prefix))
    repeat_rewards = []
    repeat_lengths = []
    for repeat in range(cfg.eval_repeat_nums):
        make_env_fn = partial(make_mdp_env, pixel=repeat < cfg.eval_record_nums)
        make_env_list = build_make_env_list(cfg.env_name, make_env_fn, oracle_context)
        eval_env = SerialEnv(len(make_env_list), make_env_list, shared_memory=False)
        if repeat < cfg.eval_record_nums:
            eval_env = TransformedEnv(eval_env, VideoRecorder(logger, log_prefix))

        rewards = torch.zeros(len(make_env_list))
        lengths = torch.zeros(len(make_env_list))
        tensordict = eval_env.reset().to(device)
        ever_done = torch.zeros(*tensordict.batch_size, 1).to(bool)

        for _ in range(cfg.env_max_steps):
            pbar.update()
            with set_interaction_mode("mode"):
                if "pixels" in tensordict.keys():
                    del tensordict["pixels"]
                action = policy(tensordict.to(device)).cpu()
                with catch_warnings():
                    filterwarnings("ignore", category=UserWarning)
                    tensordict = eval_env.step(action)

            reward = tensordict.get(("next", "reward"))
            reward[ever_done] = 0
            rewards += reward.reshape(-1)
            ever_done |= tensordict.get(("next", "done"))
            lengths += (~ever_done).float().reshape(-1)
            if ever_done.all():
                break
            else:
                tensordict = step_mdp(tensordict, exclude_action=False)
        repeat_rewards.append(rewards)
        repeat_lengths.append(lengths)

        if repeat < cfg.eval_record_nums:
            print(repeat_rewards)
            eval_env.transform.dump(suffix=str(frames_per_task))

    if logger is not None:
        logger.add_scaler("{}/eval_episode_reward".format(log_prefix), torch.stack(repeat_rewards).mean())
        logger.add_scaler("{}/eval_episode_length".format(log_prefix), torch.stack(repeat_lengths).mean())
        logger.dump_scaler(frames_per_task)

    return torch.stack(repeat_rewards)


def find_world_model(policy, task_num):
    new_policy = deepcopy(policy).to(next(policy.parameters()).device)

    sub_module = new_policy
    while not isinstance(sub_module, MDPEnv):
        if isinstance(sub_module, TensorDictModuleWrapper):
            sub_module = sub_module.td_module
        elif isinstance(sub_module, TensorDictModule):
            sub_module = sub_module.module
        else:
            raise ValueError("got {}".format(sub_module))
    new_model_env = sub_module

    new_world_model = new_model_env.world_model
    new_world_model.reset(task_num=task_num)

    return new_policy, new_world_model


def train_model(
        cfg,
        replay_buffer,
        world_model,
        world_model_loss,
        training_steps,
        model_opt,
        logits_opt=None,
        logger=None,
        deterministic_mask=False,
        log_prefix="model",
        iters=0,
        only_train=None
):
    device = next(world_model.parameters()).device
    train_logits = cfg.model_type == "causal" and cfg.reinforce and logits_opt is not None

    for step in range(training_steps):
        world_model.zero_grad()

        sampled_tensordict = replay_buffer.sample(cfg.batch_size)
        sampled_tensordict = sampled_tensordict.to(device, non_blocking=True)

        if train_logits and iters % (cfg.train_mask_iters + cfg.train_model_iters) >= cfg.train_model_iters:
            grad = world_model_loss.reinforce(sampled_tensordict, only_train)
            causal_mask = world_model.causal_mask
            logits = causal_mask.mask_logits
            logits.backward(grad)
            logits_opt.step()
            for out_dim, in_dim in product(range(logits.shape[0]), range(logits.shape[1])):
                out_name = f"o{out_dim}"
                if in_dim < causal_mask.observed_input_dim:
                    in_name = f"i{in_dim}"
                else:
                    in_name = f"c{in_dim - causal_mask.observed_input_dim}"
                logger.add_scaler(f"{log_prefix}/logits({out_name},{in_name})", logits[out_dim, in_dim])
        else:
            loss_td, all_loss = world_model_loss(sampled_tensordict, deterministic_mask, only_train)
            context_penalty = (world_model.context_model.context_hat ** 2).sum()
            all_loss += context_penalty * 0.1
            all_loss.backward()
            model_opt.step()

            if logger is not None:
                for dim in range(loss_td["transition_loss"].shape[-1]):
                    logger.add_scaler(f"{log_prefix}/obs_{dim}", loss_td["transition_loss"][..., dim].mean())
                logger.add_scaler(f"{log_prefix}/reward", loss_td["reward_loss"].mean())
                logger.add_scaler(f"{log_prefix}/terminated", loss_td["terminated_loss"].mean())
                if "mutual_info_loss" in loss_td.keys():
                    logger.add_scaler(f"{log_prefix}/mutual_info_loss", loss_td["mutual_info_loss"].mean())
                if "context_loss" in loss_td.keys():
                    logger.add_scaler(f"{log_prefix}/context", loss_td["context_loss"].mean())

        iters += 1
        # if logits_opt is None:
        #     print(world_model.context_model.context_hat.data.std(dim=0))
    return iters


def meta_test(
        cfg,
        make_env_list,
        oracle_context,
        policy,
        logger,
        frames_per_task,
        adapt_threshold=-3.
):
    logger.dump_scaler(frames_per_task)

    task_num = len(make_env_list)

    policy, world_model = find_world_model(policy, task_num=task_num)
    device = next(world_model.parameters()).device

    world_model_loss = CausalWorldModelLoss(
        world_model,
        lambda_transition=cfg.lambda_transition,
        lambda_reward=cfg.lambda_reward,
        lambda_terminated=cfg.lambda_terminated,
        lambda_mutual_info=cfg.lambda_mutual_info,
        sparse_weight=cfg.sparse_weight,
        context_sparse_weight=cfg.context_sparse_weight,
        context_max_weight=cfg.context_max_weight,
        sampling_times=cfg.sampling_times,
    ).to(device)

    collector = SyncDataCollector(
        create_env_fn=SerialEnv(len(make_env_list), make_env_list, shared_memory=False),
        policy=policy,
        total_frames=cfg.meta_task_adjust_frames_per_task * task_num,
        frames_per_batch=task_num,
        init_random_frames=0,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=ListStorage(max_size=cfg.meta_task_adjust_frames_per_task * task_num),
    )
    context_opt = torch.optim.SGD(world_model.get_parameter("context"), lr=cfg.context_lr)

    pbar = tqdm(total=cfg.meta_task_adjust_frames_per_task * task_num, desc="meta_test_adjust")
    train_model_iters = 0
    for frame, tensordict in enumerate(collector):
        pbar.update(task_num)
        replay_buffer.extend(tensordict.reshape(-1))

        train_model_iters = train_model(
            cfg, replay_buffer, world_model, world_model_loss,
            training_steps=cfg.meta_test_model_learning_per_frame * task_num,
            model_opt=context_opt,
            logger=logger,
            log_prefix=f"meta_test_model_{frames_per_task}",
            iters=train_model_iters
        )
        plot_context(cfg, world_model, oracle_context, logger, frame, log_prefix=f"meta_test_model_{frames_per_task}")
        logger.dump_scaler(frame)

    if cfg.get("new_oracle_context", None):  # adapt to target domain, only for transition
        with torch.no_grad():
            sampled_tensordict = replay_buffer.sample(10000).to(device, non_blocking=True)
            loss_td, all_loss = world_model_loss(sampled_tensordict, deterministic_mask=True)
        mean_transition_loss = loss_td["transition_loss"].mean(0)
        adapt_idx = torch.where(mean_transition_loss > adapt_threshold)[0].tolist()
        print(mean_transition_loss)
        print(adapt_idx)
        world_model.causal_mask.reset(adapt_idx)
        world_model.context_model.fix(world_model.causal_mask.valid_context_idx)

        nets_opt = torch.optim.Adam(world_model.get_parameter("nets"), lr=cfg.world_model_lr)
        model_opt = MultiOptimizer(nets=nets_opt, context=context_opt)
        logits_opt = torch.optim.Adam(world_model.get_parameter("context_logits"), lr=cfg.context_logits_lr)

        for frame in range(cfg.meta_task_adjust_frames_per_task, 5 * cfg.meta_task_adjust_frames_per_task):
            train_model_iters = train_model(
                cfg, replay_buffer, world_model, world_model_loss,
                training_steps=cfg.meta_test_model_learning_per_frame * task_num,
                model_opt=model_opt, logits_opt=logits_opt,
                logger=logger,
                log_prefix=f"meta_test_model_{frames_per_task}",
                iters=train_model_iters,
                only_train=adapt_idx
            )
            plot_context(cfg, world_model, oracle_context, logger, frame,
                         log_prefix=f"meta_test_model_{frames_per_task}")
            logger.dump_scaler(frame)
            if cfg.model_type == "causal":
                print("envs test causal mask:")
                print(world_model.causal_mask.printing_mask)

    evaluate_policy(cfg, make_env_list, policy, logger, frames_per_task, log_prefix="meta_test")


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
