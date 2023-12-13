from itertools import product
from collections import defaultdict
from functools import partial
from copy import deepcopy
import os
import math

from tqdm import tqdm
import hydra
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleWrapper
from tensordict.nn.probabilistic import set_interaction_mode
from torchrl.envs.utils import step_mdp
from torchrl.envs import TransformedEnv, SerialEnv, RewardSum, DoubleToFloat, Compose
from torchrl.envs.libs import GymEnv
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.trainers.helpers.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.modules.tensordict_module.exploration import AdditiveGaussianWrapper

from tdfa.helpers.models import make_mlp_model
from tdfa.objectives.causal_mdp import CausalWorldModelLoss
from tdfa.envs.meta_transform import MetaIdxTransform
from tdfa.envs.mdp_env import MDPEnv
from tdfa.modules.planners.cem import MyCEMPlanner as CEMPlanner

from matplotlib import pyplot as plt


class MultiOptimizer:
    def __init__(self, **optimizers):
        self.optimizers = optimizers

    def step(self):
        for opt in self.optimizers.values():
            opt.step()


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
        transforms = [DoubleToFloat(), RewardSum()]
        if idx is not None:
            transforms.append(MetaIdxTransform(idx, task_num))
        return TransformedEnv(env, transform=Compose(*transforms))

    if not cfg.meta:
        return [make_env], None
    else:
        task_num = cfg.task_num if mode == "train" else cfg.meta_test_task_num
        assert task_num >= 1

        context_dict = {}
        for key, (low, high) in cfg.oracle_context.items():
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
        log_scalar,
        max_steps=200,
        prefix="meta_train",
):
    eval_env = SerialEnv(len(make_env_list), make_env_list, shared_memory=False)
    device = next(policy.parameters()).device

    pbar = tqdm(total=cfg.eval_repeat_nums * max_steps, desc="{}_eval".format(prefix))
    repeat_rewards = []
    for repeat in range(cfg.eval_repeat_nums):
        rewards = torch.zeros(len(make_env_list), device=device)
        tensordict = eval_env.reset().to(device)
        ever_done = torch.zeros(*tensordict.batch_size, 1).to(bool).to(device)

        for _ in range(max_steps):
            pbar.update()
            with torch.no_grad() and set_interaction_mode("mode"):
                tensordict = eval_env.step(policy(tensordict).cpu()).to(device)

            reward = tensordict.get(("next", "reward"))
            reward[ever_done] = 0
            rewards += reward.reshape(-1)
            ever_done |= tensordict.get(("next", "done"))
            if ever_done.all():
                break
            else:
                tensordict = step_mdp(tensordict, exclude_action=False)
        repeat_rewards.append(rewards)

    log_scalar("{}/eval_episode_reward".format(prefix), torch.stack(repeat_rewards).mean())


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
    new_world_model.reset_context(task_num=task_num)

    return new_policy, new_world_model


def train_model(
        cfg,
        replay_buffer,
        world_model,
        world_model_loss,
        training_steps,
        model_opt,
        logits_opt=None,
        log_scalar=None
):
    device = next(world_model.parameters()).device
    train_logits = cfg.model_type == "causal" and cfg.reinforce and logits_opt is not None

    for step in range(training_steps):
        world_model.zero_grad()
        sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(device, non_blocking=True)

        if train_logits and step % (cfg.train_mask_iters + cfg.train_model_iters) < cfg.train_mask_iters:
            grad = world_model_loss.reinforce(sampled_tensordict)
            logits = world_model.causal_mask.mask_logits
            logits.backward(grad)
            logits_opt.step()
        else:
            loss_td, all_loss = world_model_loss(sampled_tensordict)
            all_loss.backward()
            model_opt.step()
            if log_scalar is not None:
                for dim in range(loss_td["transition_loss"].shape[-1]):
                    log_scalar("model_loss/obs_{}".format(dim), loss_td["transition_loss"][..., dim].mean())
                log_scalar("model_loss/reward", loss_td["reward_loss"].mean())
                log_scalar("model_loss/terminated", loss_td["terminated_loss"].mean())
                if "mutual_info_loss" in loss_td.keys():
                    log_scalar("model_loss/mutual_info_loss", loss_td["mutual_info_loss"].mean())
                if "context_loss" in loss_td.keys():
                    log_scalar("model_loss/context", loss_td["context_loss"].mean())


def meta_test(
        cfg,
        make_env_list,
        oracle_context,
        policy,
        log_scalar,
        frames_per_task
):
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

    replay_buffer = TensorDictReplayBuffer()

    context_opt = torch.optim.Adam(world_model.get_parameter("context"), lr=cfg.context_lr)

    pbar = tqdm(total=cfg.meta_task_adjust_frames_per_task * task_num, desc="meta_test_adjust")
    for frames, tensordict in enumerate(collector):
        pbar.update(task_num)
        replay_buffer.extend(tensordict.reshape(-1))
        train_model(cfg, replay_buffer, world_model, world_model_loss, model_opt=context_opt,
                    training_steps=cfg.meta_test_model_learning_per_frame * task_num)
        plot_context(cfg, world_model, oracle_context, log_scalar, frames, prefix="meta_test")

    with torch.no_grad():
        sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(device, non_blocking=True)
        loss_td, all_loss = world_model_loss(sampled_tensordict)
        for dim in range(loss_td["transition_loss"].shape[-1]):
            log_scalar("meta_test_model_loss/obs_{}".format(dim), loss_td["transition_loss"][..., dim].mean())
        log_scalar("meta_test_model_loss/reward", loss_td["reward_loss"].mean())
        log_scalar("meta_test_model_loss/terminated", loss_td["terminated_loss"].mean())

    evaluate_policy(cfg, make_env_list, policy, log_scalar, prefix="meta_test")
    plot_context(cfg, world_model, oracle_context, log_scalar, frames_per_task, prefix="meta_test")


def plot_context(
        cfg,
        world_model,
        oracle_context,
        log_scalar,
        frames_per_task,
        prefix="meta_train"
):
    context_model = world_model.context_model
    context_gt = torch.stack([v for v in oracle_context.values()], dim=-1)

    if cfg.model_type == "causal":
        valid_context_idx = world_model.causal_mask.valid_context_idx
    else:
        valid_context_idx = torch.arange(context_model.max_context_dim)
    log_scalar("{}/valid_context_num".format(prefix), float(len(valid_context_idx)))

    mcc, permutation, context_hat = context_model.get_mcc(context_gt, valid_context_idx)
    idxes_gt, idxes_hat = permutation
    log_scalar("{}/mcc".format(prefix), mcc)

    os.makedirs(prefix, exist_ok=True)

    if len(idxes_gt) == 0:
        pass
    elif len(idxes_gt) == 1:
        plt.scatter(context_gt[:, 0], context_hat[:, 0])
    else:
        num_rows = math.ceil(math.sqrt(len(idxes_gt)))
        num_cols = math.ceil(len(idxes_gt) / num_rows)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

        context_names = list(oracle_context.keys())
        for j, (idx_gt, idx_hat) in enumerate(zip(idxes_gt, idxes_hat)):
            ax = axs.flatten()[j]
            ax.set(xlabel=context_names[idx_gt], ylabel="{}th context".format(valid_context_idx[idx_hat]))
            ax.scatter(context_gt[:, idx_gt], context_hat[:, idx_hat])

        for j in range(len(idxes_gt), len(axs.flat)):
            axs.flat[j].set_visible(False)

    plt.savefig("{}/{}.png".format(prefix, frames_per_task))
    plt.close()
