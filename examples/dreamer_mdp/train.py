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
from torchrl.collectors.collectors import aSyncDataCollector

from causal_meta.utils import make_mdp_dreamer, build_logger
from causal_meta.objectives.mdp import CausalWorldModelLoss, DreamActorLoss, DreamCriticLoss
from causal_meta.utils.envs import make_mdp_env, create_make_env_list
from causal_meta.utils.eval import evaluate_policy

from utils import (
    meta_test,
    plot_context,
    match_length,
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

    make_env = partial(make_mdp_env, max_steps=cfg.env_max_steps)
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
        pred_continue=cfg.pred_continue,
        lambda_entropy=cfg.lambda_entropy
    )
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
    collector = aSyncDataCollector(
        create_env_fn=serial_env,
        policy=explore_policy,
        total_frames=cfg.train_frames_per_task,
        frames_per_batch=cfg.frames_per_batch,
        init_random_frames=cfg.init_frames_per_task,
        device=collector_device,
        storing_device=collector_device,
        split_trajs=True
    )

    # replay buffer
    buffer_size = cfg.train_frames_per_task if cfg.buffer_size == -1 else cfg.buffer_size
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(max_size=buffer_size),
    )
    final_seed = collector.set_seed(cfg.seed)
    print(f"init seed: {cfg.seed}, final seed: {final_seed}")

    # optimizers
    world_model_opt = torch.optim.Adam(world_model.get_parameter("nets"), lr=cfg.world_model_lr)
    world_model_opt.add_param_group(dict(params=world_model.get_parameter("context"), lr=cfg.context_lr))
    if cfg.model_type == "causal" and cfg.reinforce:
        logits_opt = torch.optim.Adam(world_model.get_parameter("observed_logits"), lr=cfg.observed_logits_lr)
        logits_opt.add_param_group(dict(params=world_model.get_parameter("context_logits"), lr=cfg.context_logits_lr))
    else:
        logits_opt = None
        if cfg.model_type == "causal":
            world_model_opt.add_param_group(dict(params=world_model.get_parameter("observed_logits"),
                                                 lr=cfg.observed_logits_lr))
            world_model_opt.add_param_group(dict(params=world_model.get_parameter("context_logits"),
                                                 lr=cfg.context_logits_lr))

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    # Training loop
    collected_frames = 0
    train_model_iters = 0
    pbar = tqdm(total=cfg.train_frames_per_task)
    for i, tensordict in enumerate(collector):
        current_frames = tensordict.get(("collector", "mask")).sum().item()
        pbar.update(current_frames)
        collected_frames += current_frames

        tensordict = match_length(tensordict, cfg.batch_length)
        tensordict = tensordict.reshape(-1, cfg.batch_length)

        replay_buffer.extend(tensordict)

        mask = tensordict.get(("collector", "mask"))
        episode_reward = tensordict.get(("next", "episode_reward"))[mask]
        episode_length = tensordict["next", "step_count"][mask].float()
        done = tensordict.get(("next", "done"))[mask]
        logger.add_scaler("rollout/reward_mean", tensordict[("next", "reward")][mask].mean())
        logger.add_scaler("rollout/reward_std", tensordict[("next", "reward")][mask].std())
        if done.any():
            logger.add_scaler("rollout/episode_reward", episode_reward[done].mean())
            logger.add_scaler("rollout/episode_length", episode_length[done].mean())
        logger.add_scaler("rollout/action_mean", tensordict["action"][mask].mean())
        logger.add_scaler("rollout/action_std", tensordict["action"][mask].std())
        logger.dump_scaler(collected_frames)

        if collected_frames < cfg.init_frames_per_task:
            continue

        train_model_iters = train_model(
            cfg, replay_buffer, world_model, world_model_loss,
            cfg.optim_steps_per_frame, world_model_opt, logits_opt, logger,
            iters=train_model_iters
        )

        train_policy(cfg, replay_buffer, actor_loss, critic_loss, cfg.optim_steps_per_frame,
                     actor_opt, critic_opt, logger)

        if (i + 1) % cfg.eval_interval == 0:
            evaluate_policy(cfg, train_oracle_context, explore_policy, logger, collected_frames)

        # if cfg.meta and (i + 1) % cfg.meta_test_interval_frames_per_task == 0:
        #     meta_test(cfg, test_make_env_list, test_oracle_context, explore_policy, logger, collected_frames)

        if cfg.meta:
            plot_context(cfg, world_model, train_oracle_context, logger, collected_frames)
        if cfg.model_type == "causal":
            print()
            print(world_model.causal_mask.printing_mask)
        logger.dump_scaler(collected_frames)

        if (i + 1) % cfg.save_model_interval == 0:
            os.makedirs("world_model", exist_ok=True)
            torch.save(world_model.state_dict(), os.path.join(f"world_model/{collected_frames}.pt"))

    collector.shutdown()


if __name__ == '__main__':
    main()
