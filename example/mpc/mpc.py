from itertools import product
from collections import defaultdict
from functools import partial
import os
import math

from tqdm import tqdm
import hydra
import torch
from tensordict import TensorDict
from torchrl.envs.utils import step_mdp
from torchrl.envs import TransformedEnv, SerialEnv, RewardSum, DoubleToFloat, Compose
from torchrl.envs.libs import GymEnv
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.trainers.helpers.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.modules.tensordict_module.exploration import AdditiveGaussianWrapper
from matplotlib import pyplot as plt

from tdfa.helpers.models import make_mlp_model
from tdfa.objectives.causal_mdp import CausalWorldModelLoss
from tdfa.envs.meta_transform import MetaIdxTransform
from tdfa.modules.planners.cem import MyCEMPlanner as CEMPlanner

from utils import env_constructor, get_dim_map, evaluate_policy, meta_test, plot_context, MultiOptimizer, train_model


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg):
    if torch.cuda.is_available():
        device = torch.device(cfg.device)
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    exp_name = generate_exp_name("MPC", cfg.exp_name)
    logger = get_logger(
        logger_type=cfg.logger,
        logger_name="mpc",
        experiment_name=exp_name,
        wandb_kwargs={
            "project": "causal_rl",
            "group": f"MPC_{cfg.env_name}",
            "offline": cfg.offline_logging,
        },
    )

    train_make_env_list, train_oracle_context = env_constructor(cfg, mode="train")
    test_make_env_list, test_oracle_context = env_constructor(cfg, mode="test")

    task_num = len(train_make_env_list)
    proof_env = train_make_env_list[0]()
    world_model, model_env = make_mlp_model(cfg, proof_env, device=device)

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

    collector = SyncDataCollector(
        create_env_fn=SerialEnv(task_num, train_make_env_list, shared_memory=False),
        policy=explore_policy,
        total_frames=cfg.train_frames_per_task * task_num,
        frames_per_batch=task_num,
        init_random_frames=cfg.init_frames_per_task * task_num,
    )

    replay_buffer = TensorDictReplayBuffer()

    context_opt = torch.optim.Adam(world_model.get_parameter("context"), lr=cfg.context_lr)
    module_opt = torch.optim.Adam(world_model.get_parameter("module"), lr=cfg.world_model_lr)
    model_opt = MultiOptimizer(module=module_opt, context=context_opt)
    if cfg.model_type == "causal":
        logits_opt = MultiOptimizer(
            observed_logits=torch.optim.Adam(world_model.get_parameter("observed_logits"), lr=cfg.observed_logits_lr),
            context_logits=torch.optim.Adam(world_model.get_parameter("context_logits"), lr=cfg.context_logits_lr)
        )
    else:
        logits_opt = None

    input_dim_map, output_dim_map = get_dim_map(
        obs_dim=proof_env.observation_spec["observation"].shape[0],
        action_dim=proof_env.action_spec.shape[0],
        context_dim=cfg.max_context_dim if cfg.meta else 0
    )
    del proof_env

    logging_scalar = defaultdict(list)

    def log_scalar(key, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        logging_scalar[key].append(value)

    pbar = tqdm(total=cfg.train_frames_per_task * task_num)
    collected_frames = 0
    model_learning_steps = 0
    for frames_per_task, tensordict in enumerate(collector):
        pbar.update(task_num)
        collected_frames += task_num

        if tensordict["next", "done"].any():
            episode_reward = tensordict["next", "episode_reward"][tensordict["next", "done"]]
            log_scalar("meta_train/rollout_episode_reward", episode_reward.mean())

        replay_buffer.extend(tensordict.reshape(-1))

        if frames_per_task < cfg.init_frames_per_task:
            continue

        train_model(cfg, replay_buffer, world_model, world_model_loss,
                    cfg.model_learning_per_frame * task_num, model_opt, logits_opt, log_scalar)
        if frames_per_task % cfg.reset_context_frames_per_task == 0:
            world_model.reset_context()
            train_model(cfg, replay_buffer, world_model, world_model_loss,
                        cfg.model_learning_per_frame * task_num, context_opt)

        if cfg.model_type == "causal":
            logits = world_model.causal_mask.mask_logits
            for out_dim, in_dim in product(range(logits.shape[0]), range(logits.shape[1])):
                log_scalar(
                    "mask_logits/{},{}".format(output_dim_map(out_dim), input_dim_map(in_dim)),
                    logits[out_dim, in_dim],
                )

        # if frames_per_task % cfg.eval_interval_frames_per_task == 0:
        #     evaluate_policy(cfg, train_make_env_list, explore_policy, log_scalar)
        #
        # if frames_per_task % cfg.meta_test_interval_frames_per_task == 0:
        #     meta_test(cfg, test_make_env_list, test_oracle_context, explore_policy, log_scalar, frames_per_task)

        if frames_per_task % cfg.record_interval_frames_per_task == 0:
            if cfg.meta:
                plot_context(cfg, world_model, train_oracle_context, log_scalar, frames_per_task)

            for key, value in logging_scalar.items():
                if len(value) == 0:
                    continue
                if len(value) > 1:
                    value = torch.tensor(value).mean()
                else:
                    value = value[0]

                if logger is not None:
                    logger.log_scalar(key, value, step=collected_frames)
                else:
                    print(f"{key}: {value}")

            if cfg.model_type == "causal":
                print()
                print(world_model.causal_mask.printing_mask)

            logging_scalar = defaultdict(list)

    collector.shutdown()


if __name__ == '__main__':
    main()
