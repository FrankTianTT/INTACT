from itertools import product
from collections import defaultdict
from functools import partial
from copy import deepcopy
import os
import math
from warnings import catch_warnings, filterwarnings

import numpy as np
from tqdm import tqdm
import hydra
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleWrapper
from tensordict.nn.probabilistic import set_interaction_mode
from torchrl.envs.utils import step_mdp
from torchrl.envs import TransformedEnv, SerialEnv, RewardSum, DoubleToFloat, Compose, StepCounter
from torchrl.envs.libs import GymEnv
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.trainers.helpers.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import ListStorage
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import cm

from causal_meta.helpers.models import make_mdp_model
from causal_meta.objectives.causal_mdp import CausalWorldModelLoss
from causal_meta.envs.meta_transform import MetaIdxTransform
from causal_meta.envs.mdp_env import MDPEnv
from causal_meta.modules.planners.cem import MyCEMPlanner as CEMPlanner


class MultiOptimizer:
    def __init__(self, **optimizers):
        self.optimizers = optimizers

    def step(self):
        for opt in self.optimizers.values():
            opt.step()


class MyLogger:
    def __init__(self, cfg, name="mpc", log_dir=""):
        exp_name = generate_exp_name(name.upper(), cfg.exp_name)
        if log_dir == "":
            log_dir = os.path.join(log_dir, name)

        self.logger = get_logger(
            logger_type=cfg.logger,
            logger_name=log_dir,
            experiment_name=exp_name,
            wandb_kwargs={
                "project": "causal_rl",
                "group": f"MPC_{cfg.env_name}",
                "config": dict(cfg),
                "offline": cfg.offline_logging,
            },
        )
        self._cache = defaultdict(list)

    def log_scalar(self, key, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        self._cache[key].append(value)

    def update(self, step):
        for key, value in self._cache.items():
            if len(value) == 0:
                continue
            if len(value) > 1:
                value = torch.tensor(value).mean()
            else:
                value = value[0]

            self.logger.log_scalar(key, value, step=step)

        self._cache = defaultdict(list)


def get_dim_map(obs_dim, action_dim, context_dim):
    def input_dim_map(dim):
        assert dim < obs_dim + action_dim + context_dim
        if dim < obs_dim:
            return "obs_{}".format(dim)
        elif dim < obs_dim + action_dim:
            return "action_{}".format(dim - obs_dim)
        else:
            return "context_{}".format(dim - obs_dim - action_dim)

    def output_dim_map(dim):
        assert dim < obs_dim + 2
        if dim < obs_dim:
            return "obs_{}".format(dim)
        elif dim < obs_dim + 1:
            return "reward"
        else:
            return "terminated"

    return input_dim_map, output_dim_map


def env_constructor(cfg, mode="train"):
    def make_env(gym_kwargs=None, idx=None, task_num=None):
        if gym_kwargs is None:
            gym_kwargs = {}
        env = GymEnv(cfg.env_name, **gym_kwargs)
        transforms = [DoubleToFloat(), RewardSum(), StepCounter()]
        if idx is not None:
            transforms.append(MetaIdxTransform(idx, task_num))
        return TransformedEnv(env, transform=Compose(*transforms))

    if not cfg.meta:
        return [make_env], None
    else:
        task_num = cfg.task_num if mode == "meta_train" else cfg.meta_test_task_num
        assert task_num >= 1

        oracle_context = dict(cfg.oracle_context)
        if mode == "meta_test" and cfg.get("new_oracle_context", None):
            oracle_context.update(dict(cfg.new_oracle_context))
        context_dict = {}
        for key, (low, high) in oracle_context.items():
            context_dict[key] = torch.rand(task_num) * (high - low) + low
        oracle_context = TensorDict(context_dict, batch_size=task_num)

        make_env_list = []
        for idx in range(task_num):
            gym_kwargs = dict([(key, value[idx].item()) for key, value in context_dict.items()])
            make_env_list.append(partial(make_env, gym_kwargs=gym_kwargs, idx=idx))
        return make_env_list, oracle_context


def evaluate_policy(
        cfg,
        make_env_list,
        policy,
        logger=None,
        log_prefix="meta_train",
):
    eval_env = SerialEnv(len(make_env_list), make_env_list, shared_memory=False)
    device = next(policy.parameters()).device

    pbar = tqdm(total=cfg.eval_repeat_nums * cfg.env_max_steps, desc="{}_eval".format(log_prefix))
    repeat_rewards = []
    repeat_lengths = []
    for repeat in range(cfg.eval_repeat_nums):
        rewards = torch.zeros(len(make_env_list), device=device)
        lengths = torch.zeros(len(make_env_list), device=device)
        tensordict = eval_env.reset().to(device)
        ever_done = torch.zeros(*tensordict.batch_size, 1).to(bool).to(device)

        for _ in range(cfg.env_max_steps):
            pbar.update()
            with set_interaction_mode("mode"):
                action = policy(tensordict).cpu()
                with catch_warnings():
                    filterwarnings("ignore", category=UserWarning)
                    tensordict = eval_env.step(action).to(device)

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

    if logger is not None:
        logger.log_scalar("{}/eval_episode_reward".format(log_prefix), torch.stack(repeat_rewards).mean())
        logger.log_scalar("{}/eval_episode_length".format(log_prefix), torch.stack(repeat_lengths).mean())

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


from time import time


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
                logger.log_scalar(f"{log_prefix}/logits({out_name},{in_name})", logits[out_dim, in_dim])
        else:
            loss_td, all_loss = world_model_loss(sampled_tensordict, deterministic_mask, only_train)
            context_penalty = (world_model.context_model.context_hat ** 2).sum()
            all_loss += context_penalty * 0.1
            all_loss.backward()
            model_opt.step()

            if logger is not None:
                for dim in range(loss_td["transition_loss"].shape[-1]):
                    logger.log_scalar(f"{log_prefix}/obs_{dim}", loss_td["transition_loss"][..., dim].mean())
                logger.log_scalar(f"{log_prefix}/reward", loss_td["reward_loss"].mean())
                logger.log_scalar(f"{log_prefix}/terminated", loss_td["terminated_loss"].mean())
                if "mutual_info_loss" in loss_td.keys():
                    logger.log_scalar(f"{log_prefix}/mutual_info_loss", loss_td["mutual_info_loss"].mean())
                if "context_loss" in loss_td.keys():
                    logger.log_scalar(f"{log_prefix}/context", loss_td["context_loss"].mean())

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
    logger.update(frames_per_task)

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
        logger.update(frame)

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

        module_opt = torch.optim.Adam(world_model.get_parameter("module"), lr=cfg.world_model_lr)
        model_opt = MultiOptimizer(module=module_opt, context=context_opt)
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
            logger.update(frame)
            if cfg.model_type == "causal":
                print("meta test causal mask:")
                print(world_model.causal_mask.printing_mask)

    evaluate_policy(cfg, make_env_list, policy, logger, log_prefix="meta_test")


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
        logger.log_scalar("{}/valid_context_num".format(log_prefix), float(len(valid_context_idx)))
        logger.log_scalar("{}/mcc".format(log_prefix), mcc)


@hydra.main(version_base="1.1", config_path="", config_name="config")
def test_env_constructor(cfg):
    train_make_env_list, train_oracle_context = env_constructor(cfg, mode="meta_train")
    test_make_env_list, test_oracle_context = env_constructor(cfg, mode="meta_test")


if __name__ == '__main__':
    test_env_constructor()
