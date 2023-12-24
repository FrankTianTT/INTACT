from itertools import product
from collections import defaultdict
from functools import partial
import os
import time
import math

from tqdm import tqdm
import hydra
import torch
from tensordict import TensorDict
from torchrl.envs.utils import step_mdp
from torchrl.envs import TransformedEnv, SerialEnv, RewardSum, DoubleToFloat, Compose
from torchrl.envs.libs import GymEnv
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.trainers.helpers.collectors import MultiaSyncDataCollector, SyncDataCollector
from torchrl.data.replay_buffers.storages import ListStorage
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.modules.tensordict_module.exploration import AdditiveGaussianWrapper
from matplotlib import pyplot as plt

from causal_meta.helpers import make_mdp_model, build_logger
from causal_meta.objectives.causal_mdp import CausalWorldModelLoss
from causal_meta.envs.meta_transform import MetaIdxTransform
from causal_meta.modules.planners.cem import MyCEMPlanner as CEMPlanner

from utils import (
    env_constructor,
    build_make_env_list,
    evaluate_policy,
    meta_test,
    plot_context,
    MultiOptimizer,
    train_model
)


@hydra.main(version_base="1.1", config_path="conf", config_name="main")
def main(cfg):
    if torch.cuda.is_available():
        device = torch.device(cfg.device)
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    logger = build_logger(cfg)

    train_make_env_list, train_oracle_context = env_constructor(cfg, mode="meta_train")
    test_make_env_list, test_oracle_context = env_constructor(cfg, mode="meta_test")
    torch.save(train_oracle_context, "train_oracle_context.pt")
    torch.save(test_oracle_context, "test_oracle_context.pt")
    print("train_make_env_list", train_make_env_list)

    task_num = len(train_make_env_list)
    proof_env = train_make_env_list[0]()
    world_model, model_env = make_mdp_model(cfg, proof_env, device=device)

    world_model_loss = CausalWorldModelLoss(
        world_model,
        lambda_transition=cfg.lambda_transition,
        lambda_reward=cfg.lambda_reward if cfg.reward_fns == "" else 0.,
        lambda_terminated=cfg.lambda_terminated if cfg.termination_fns == "" else 0.,
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
        sigma_init=0.3,
        sigma_end=0.3,
        spec=proof_env.action_spec,
    )
    del proof_env

    collector = SyncDataCollector(
        # create_env_fn=[lambda: SerialEnv(task_num, train_make_env_list, shared_memory=False)],
        create_env_fn=SerialEnv(task_num, train_make_env_list, shared_memory=False),
        policy=explore_policy,
        total_frames=cfg.train_frames_per_task * task_num,
        frames_per_batch=task_num,
        init_random_frames=cfg.init_frames_per_task * task_num,
        device=cfg.collector_device,
        storing_device=cfg.collector_device,
    )

    buffer_size = cfg.train_frames_per_task * task_num if cfg.buffer_size == -1 else cfg.buffer_size
    replay_buffer = TensorDictReplayBuffer(
        storage=ListStorage(max_size=buffer_size),
    )

    context_opt = torch.optim.SGD(world_model.get_parameter("context"), lr=cfg.context_lr)
    module_opt = torch.optim.Adam(world_model.get_parameter("module"), lr=cfg.world_model_lr,
                                  weight_decay=cfg.world_model_weight_decay)
    model_opt = MultiOptimizer(module=module_opt, context=context_opt)
    if cfg.model_type == "causal":
        logits_opt = MultiOptimizer(
            observed_logits=torch.optim.Adam(world_model.get_parameter("observed_logits"), lr=cfg.observed_logits_lr),
            context_logits=torch.optim.Adam(world_model.get_parameter("context_logits"), lr=cfg.context_logits_lr)
        )
    else:
        logits_opt = None

    pbar = tqdm(total=cfg.train_frames_per_task * task_num)
    train_model_iters = 0
    for frames_per_task, tensordict in enumerate(collector):
        pbar.update(task_num)

        if tensordict["next", "done"].any():
            episode_reward = tensordict["next", "episode_reward"][tensordict["next", "done"]]
            episode_length = tensordict["next", "step_count"][tensordict["next", "done"]].float()
            logger.add_scaler("meta_train/rollout_episode_reward", episode_reward.mean())
            logger.add_scaler("meta_train/rollout_episode_length", episode_length.mean())

        replay_buffer.extend(tensordict.reshape(-1))

        if frames_per_task < cfg.init_frames_per_task:
            continue

        train_model_iters = train_model(
            cfg, replay_buffer, world_model, world_model_loss,
            cfg.model_learning_per_frame * task_num, model_opt, logits_opt, logger,
            iters=train_model_iters
        )

        if (frames_per_task + 1) % cfg.eval_interval_frames_per_task == 0:
            evaluate_policy(cfg, train_oracle_context, explore_policy, logger, frames_per_task)

        if cfg.meta and (frames_per_task + 1) % cfg.meta_test_interval_frames_per_task == 0:
            meta_test(cfg, test_make_env_list, test_oracle_context, explore_policy, logger, frames_per_task)

        if cfg.meta:
            plot_context(cfg, world_model, train_oracle_context, logger, frames_per_task)
        if cfg.model_type == "causal":
            print()
            print(world_model.causal_mask.printing_mask)
        logger.dump_scaler(frames_per_task)

        if (frames_per_task + 1) % cfg.save_model_frames_per_task == 0:
            os.makedirs("world_model", exist_ok=True)
            torch.save(world_model.state_dict(), os.path.join(f"world_model/{frames_per_task}.pt"))

    collector.shutdown()


if __name__ == '__main__':
    main()
