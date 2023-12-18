from itertools import product
from collections import defaultdict
from functools import partial
import os
import math
import argparse

from tqdm import tqdm
import hydra
from omegaconf import OmegaConf
import torch
from tensordict import TensorDict
from torchrl.envs.utils import step_mdp
from torchrl.envs import TransformedEnv, SerialEnv, RewardSum, DoubleToFloat, Compose
from torchrl.envs.libs import GymEnv
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.trainers.helpers.collectors import SyncDataCollector
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.modules.tensordict_module.exploration import AdditiveGaussianWrapper
from matplotlib import pyplot as plt

from causal_meta.helpers.models import make_mlp_model
from causal_meta.objectives.causal_mdp import CausalWorldModelLoss
from causal_meta.envs.meta_transform import MetaIdxTransform
from causal_meta.modules.planners.cem import MyCEMPlanner as CEMPlanner

from utils import (
    env_constructor,
    get_dim_map,
    evaluate_policy,
    meta_test,
    plot_context,
    MultiOptimizer,
    MyLogger,
    train_model
)


def restore_make_env_list(cfg, oracle_context):
    def make_env(gym_kwargs=None, idx=None, task_num=None):
        if gym_kwargs is None:
            gym_kwargs = {}
        env = GymEnv(cfg.env_name, **gym_kwargs)
        transforms = [DoubleToFloat(), RewardSum()]
        if idx is not None:
            transforms.append(MetaIdxTransform(idx, task_num))
        return TransformedEnv(env, transform=Compose(*transforms))

    make_env_list = []
    for idx in range(oracle_context.shape[0]):
        gym_kwargs = dict([(key, value.item()) for key, value in oracle_context[idx].items()])
        make_env_list.append(partial(make_env, gym_kwargs=gym_kwargs, idx=idx))
    return make_env_list


def main(path, load_frames, train_frames_per_task):
    cfg_path = os.path.join(path, ".hydra", "config.yaml")
    cfg = OmegaConf.load(cfg_path)

    if torch.cuda.is_available():
        device = torch.device(cfg.device)
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    logger = MyLogger(cfg, log_dir=os.path.join(path, "load_model"))

    train_oracle_context = torch.load(os.path.join(path, "train_oracle_context.pt"), map_location=device)
    test_oracle_context = torch.load(os.path.join(path, "test_oracle_context.pt"), map_location=device)
    train_make_env_list = restore_make_env_list(cfg, train_oracle_context)
    test_make_env_list = restore_make_env_list(cfg, test_oracle_context)

    task_num = len(train_make_env_list)

    proof_env = train_make_env_list[0]()
    world_model, model_env = make_mlp_model(cfg, proof_env, device=device)
    world_model.load_state_dict(torch.load(os.path.join(path, "world_model", f"{load_frames}.pt"), map_location=device))

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

    planner = CEMPlanner(
        model_env,
        planning_horizon=cfg.planning_horizon,
        optim_steps=cfg.optim_steps,
        num_candidates=cfg.num_candidates,
        top_k=cfg.top_k,
    )

    explore_policy = AdditiveGaussianWrapper(
        planner,
        sigma_init=0.5,
        sigma_end=0.5,
        spec=proof_env.action_spec,
    )
    del proof_env

    if train_frames_per_task == -1:
        train_frames_per_task = cfg.meta_task_adjust_frames_per_task

    collector = SyncDataCollector(
        create_env_fn=SerialEnv(task_num, train_make_env_list, shared_memory=False),
        policy=explore_policy,
        total_frames=train_frames_per_task * task_num,
        frames_per_batch=task_num,
        init_random_frames=0,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(max_size=cfg.train_frames_per_task * task_num),
    )
    context_opt = torch.optim.Adam(world_model.get_parameter("context"), lr=cfg.context_lr)
    module_opt = torch.optim.Adam(world_model.get_parameter("module"), lr=cfg.world_model_lr)
    model_opt = MultiOptimizer(module=module_opt, context=context_opt)

    cfg.eval_repeat_nums = 10
    repeat_rewards = evaluate_policy(cfg, train_make_env_list, explore_policy)
    print(repeat_rewards)
    print("mean", repeat_rewards.mean(dim=0))
    print("std", repeat_rewards.std(dim=0))
    print("max", repeat_rewards.max(dim=0))
    print("min", repeat_rewards.min(dim=0))
    plot_context(
        cfg, world_model, train_oracle_context,
        plot_path=os.path.join(path, "load_model", "before_context.png"),
        color_values=repeat_rewards.mean(dim=0).cpu().numpy()
    )
    #
    # world_model.reset()
    # # plot_context(
    # #     cfg, world_model, train_oracle_context,
    # #     plot_path=os.path.join(path, "load_model", "before_context.png"),
    # #     # color_values=repeat_rewards.mean(dim=0).cpu().numpy()
    # # )
    # # repeat_rewards = evaluate_policy(cfg, train_make_env_list, explore_policy)
    #
    # print(world_model.causal_mask.printing_mask)
    #
    pbar = tqdm(total=train_frames_per_task * task_num)
    for frames_per_task, tensordict in enumerate(collector):
        pbar.update(task_num)
        if tensordict["next", "done"].any():
            episode_reward = tensordict["next", "episode_reward"][tensordict["next", "done"]]
            logger.log_scalar("meta_train/rollout_episode_reward", episode_reward.mean())

        replay_buffer.extend(tensordict.reshape(-1))

        # train_model(
        #     cfg, replay_buffer, world_model, world_model_loss,
        #     cfg.model_learning_per_frame * task_num, model_opt,
        #     logger=logger,
        #     deterministic_mask=True
        # )

        # with torch.no_grad():
        #     sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(device, non_blocking=True)
        #     loss_td, all_loss = world_model_loss(sampled_tensordict, deterministic_mask=True)
        #     log_prefix = "model"
        #     for dim in range(loss_td["transition_loss"].shape[-1]):
        #         logger.log_scalar(f"{log_prefix}/obs_{dim}", loss_td["transition_loss"][..., dim].mean())
        #     logger.log_scalar(f"{log_prefix}/reward", loss_td["reward_loss"].mean())
        #     logger.log_scalar(f"{log_prefix}/terminated", loss_td["terminated_loss"].mean())
        #     if "mutual_info_loss" in loss_td.keys():
        #         logger.log_scalar(f"{log_prefix}/mutual_info_loss", loss_td["mutual_info_loss"].mean())
        #     if "context_loss" in loss_td.keys():
        #         logger.log_scalar(f"{log_prefix}/context", loss_td["context_loss"].mean())

        logger.update(frames_per_task)

    # repeat_rewards = evaluate_policy(cfg, train_make_env_list, explore_policy)
    plot_context(
        cfg, world_model, train_oracle_context,
        plot_path=os.path.join(path, "load_model", "after_context.png"),
        # color_values=repeat_rewards.mean(dim=0).cpu().numpy()
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('load_frames', type=int)
    parser.add_argument('--train_frames_per_task', type=int, default=-1)

    args = parser.parse_args()

    main(args.path, args.load_frames, args.train_frames_per_task)
