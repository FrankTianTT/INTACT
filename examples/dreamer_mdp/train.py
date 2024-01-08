import os
from functools import partial
from time import time
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import hydra
import torch
from torchrl.envs import SerialEnv
from torchrl.trainers.helpers.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.modules.tensordict_module.exploration import AdditiveGaussianWrapper

from causal_meta.helpers import make_mdp_dreamer, build_logger
from causal_meta.objectives.mdp import CausalWorldModelLoss, DreamActorLoss, DreamCriticLoss
from causal_meta.helpers.envs import make_mdp_env, create_make_env_list

from utils import (
    evaluate_policy,
    meta_test,
    plot_context,
    MultiOptimizer,
    train_model,
    train_policy
)


@hydra.main(version_base="1.1", config_path="conf", config_name="main")
def main(cfg):
    if torch.cuda.is_available():
        device = torch.device(cfg.model_device)
        collector_device = torch.device(cfg.collector_device)
    else:
        device = torch.device("cpu")
        collector_device = torch.device("cpu")
    print(f"Using device {device}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger = build_logger(cfg, name="dreamer_mdp")

    make_env = partial(make_mdp_env, max_steps=cfg.max_steps)
    train_make_env_list, train_oracle_context = create_make_env_list(cfg, make_env, mode="meta_train")
    test_make_env_list, test_oracle_context = create_make_env_list(cfg, make_env, mode="meta_test")
    torch.save(train_oracle_context, "train_oracle_context.pt")
    torch.save(test_oracle_context, "test_oracle_context.pt")
    print("train_make_env_list", train_make_env_list)

    task_num = len(train_make_env_list)
    proof_env = train_make_env_list[0]()
    world_model, model_based_env, actor, critic = make_mdp_dreamer(cfg, proof_env, device=device)

    world_model_loss = CausalWorldModelLoss(
        world_model,
        lambda_transition=cfg.lambda_transition,
        lambda_reward=cfg.lambda_reward if cfg.reward_fns == "" else 0.,
        lambda_terminated=cfg.lambda_terminated if cfg.termination_fns == "" else 0.,
        sparse_weight=cfg.sparse_weight,
        context_sparse_weight=cfg.context_sparse_weight,
        context_max_weight=cfg.context_max_weight,
        sampling_times=cfg.sampling_times,
    ).to(device)
    actor_loss = DreamActorLoss(
        actor,
        critic,
        model_based_env,
        imagination_horizon=cfg.imagination_horizon,
        discount_loss=cfg.discount_loss,
        pred_continue=cfg.pred_continue, )
    critic_loss = DreamCriticLoss(
        critic,
        discount_loss=cfg.discount_loss,
    )

    explore_policy = AdditiveGaussianWrapper(
        actor,
        sigma_init=0.3,
        sigma_end=0.3,
        # annealing_num_steps=cfg.train_frames_per_task,
        spec=proof_env.action_spec,
    )
    del proof_env

    serial_env = SerialEnv(task_num, train_make_env_list, shared_memory=False)
    serial_env.set_seed(cfg.seed)
    collector = SyncDataCollector(
        create_env_fn=serial_env,
        policy=explore_policy,
        total_frames=cfg.train_frames_per_task * task_num,
        frames_per_batch=task_num,
        init_random_frames=cfg.init_frames_per_task * task_num,
        device=collector_device,
        storing_device=collector_device,
    )

    # replay buffer
    buffer_size = cfg.train_frames_per_task * task_num if cfg.buffer_size == -1 else cfg.buffer_size
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(max_size=buffer_size),
    )
    final_seed = collector.set_seed(cfg.seed)
    print(f"init seed: {cfg.seed}, final seed: {final_seed}")

    # optimizers
    context_opt = torch.optim.SGD(world_model.get_parameter("context"), lr=cfg.context_lr)
    nets_opt = torch.optim.Adam(world_model.get_parameter("nets"), lr=cfg.world_model_lr,
                                weight_decay=cfg.world_model_weight_decay)
    model_opt = MultiOptimizer(nets=nets_opt, context=context_opt)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    if cfg.model_type == "causal":
        logits_opt = MultiOptimizer(
            observed_logits=torch.optim.Adam(world_model.get_parameter("observed_logits"), lr=cfg.observed_logits_lr),
            context_logits=torch.optim.Adam(world_model.get_parameter("context_logits"), lr=cfg.context_logits_lr)
        )
    else:
        logits_opt = None

    # Training loop
    pbar = tqdm(total=cfg.train_frames_per_task * task_num)
    train_model_iters = 0
    t0 = time()
    for frames_per_task, tensordict in enumerate(collector):
        t1 = time()
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
            cfg.optim_steps_per_frame * task_num, model_opt, logits_opt, logger,
            iters=train_model_iters
        )

        train_policy(cfg, replay_buffer, actor_loss, critic_loss, cfg.optim_steps_per_frame * task_num,
                     actor_opt, critic_opt, logger)

        t2 = time()

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

        # print(f"collect: {t1 - t0:.2f}, train: {t2 - t1:.2f}")
        t0 = time()
    collector.shutdown()


if __name__ == '__main__':
    main()
