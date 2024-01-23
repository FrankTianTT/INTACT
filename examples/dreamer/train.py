from functools import partial
from itertools import product
import os

import numpy as np
import hydra
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tensordict.nn.probabilistic import InteractionType
from torchrl.envs import ParallelEnv, SerialEnv, TransformedEnv
from torchrl.record import VideoRecorder
from torchrl.modules.tensordict_module.exploration import AdditiveGaussianWrapper
from torchrl.objectives.dreamer import DreamerActorLoss, DreamerValueLoss
from torchrl.collectors.collectors import aSyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.trainers.trainers import Recorder, RewardNormalizer

from causal_meta.utils import make_dreamer, build_logger, evaluate_policy, plot_context, match_length
from causal_meta.utils.envs import make_dreamer_env, create_make_env_list
from causal_meta.objectives.causal_dreamer import CausalDreamerModelLoss

from utils import meta_test, train_model, train_agent


@hydra.main(version_base="1.1", config_path="conf", config_name="main")
def main(cfg: "DictConfig"):  # noqa: F821
    torch.multiprocessing.set_sharing_strategy('file_system')

    if torch.cuda.is_available():
        device = torch.device(cfg.model_device)
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger = build_logger(cfg, "dreamer")

    make_env_fn = partial(
        make_dreamer_env,
        variable_num=cfg.variable_num,
        state_dim_per_variable=cfg.state_dim_per_variable,
        hidden_dim_per_variable=cfg.belief_dim_per_variable
    )
    train_make_env_list, train_oracle_context = create_make_env_list(cfg, make_env_fn, mode="meta_train")
    test_make_env_list, test_oracle_context = create_make_env_list(cfg, make_env_fn, mode="meta_test")
    torch.save(train_oracle_context, "train_oracle_context.pt")
    torch.save(test_oracle_context, "test_oracle_context.pt")

    task_num = len(train_make_env_list)
    proof_env = train_make_env_list[0]()
    world_model, model_based_env, actor_model, value_model, policy = make_dreamer(
        cfg=cfg,
        proof_environment=proof_env,
        device=device,
    )

    if cfg.normalize_rewards_online:
        reward_normalizer = RewardNormalizer()
    else:
        reward_normalizer = None

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
    )
    actor_loss = DreamerActorLoss(
        # AdditiveGaussianWrapper(actor_model, sigma_init=0.3, sigma_end=0.3).to(device),
        actor_model,
        value_model,
        model_based_env,
        imagination_horizon=cfg.imagination_horizon,
        discount_loss=cfg.discount_loss,
        pred_continue=cfg.pred_continue,
    )
    value_loss = DreamerValueLoss(
        value_model,
        discount_loss=cfg.discount_loss,
    )

    exploration_policy = AdditiveGaussianWrapper(
        policy,
        sigma_init=0.3,
        sigma_end=0.3,
        # annealing_num_steps=cfg.train_frames_per_task * task_num,
    ).to(device)

    collector = aSyncDataCollector(
        create_env_fn=SerialEnv(task_num, train_make_env_list, shared_memory=False),
        policy=exploration_policy,
        total_frames=cfg.train_frames_per_task,
        frames_per_batch=cfg.frames_per_batch,
        init_random_frames=cfg.init_frames_per_task,
        device=cfg.collector_device,
        storing_device=cfg.collector_device,
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

    actor_opt = torch.optim.Adam(actor_model.parameters(), lr=cfg.actor_value_lr)
    value_opt = torch.optim.Adam(value_model.parameters(), lr=cfg.actor_value_lr)

    # Training loop
    collected_frames = 0
    train_model_iters = 0
    pbar = tqdm(total=cfg.train_frames_per_task)
    for i, tensordict in enumerate(collector):
        if reward_normalizer is not None:
            reward_normalizer.update_reward_stats(tensordict)

        current_frames = tensordict.get(("collector", "mask")).sum().item()
        pbar.update(current_frames)
        collected_frames += current_frames

        tensordict = match_length(tensordict, cfg.batch_length)
        tensordict = tensordict.reshape(-1, cfg.batch_length)

        replay_buffer.extend(tensordict.cpu())

        mask = tensordict.get(("collector", "mask"))
        episode_reward = tensordict.get(("next", "episode_reward"))[mask]
        done = tensordict.get(("next", "done"))[mask]
        mean_episode_reward = episode_reward[done].mean()
        logger.add_scaler("rollout/reward_mean", tensordict[("next", "reward")][mask].mean())
        logger.add_scaler("rollout/reward_std", tensordict[("next", "reward")][mask].std())
        logger.add_scaler("rollout/episode_reward_mean", mean_episode_reward)
        logger.add_scaler("rollout/action_mean", tensordict["action"][mask].mean())
        logger.add_scaler("rollout/action_std", tensordict["action"][mask].std())
        logger.dump_scaler(collected_frames)

        if collected_frames < cfg.init_frames_per_task:
            continue

        if cfg.model_type == "causal":
            print()
            print(world_model.causal_mask.printing_mask)

        train_model_iters = train_model(
            cfg, replay_buffer, world_model, world_model_loss, world_model_opt,
            cfg.optim_steps_per_batch, logits_opt, logger,
            iters=train_model_iters, reward_normalizer=reward_normalizer
        )

        if collected_frames < cfg.policy_learning_frames_per_task:
            continue

        train_agent(cfg, replay_buffer, actor_model, actor_loss, actor_opt, value_model, value_loss, value_opt,
                    cfg.optim_steps_per_batch, logger)

        if (i + 1) % cfg.eval_interval == 0:
            evaluate_policy(cfg, train_oracle_context, exploration_policy, logger, collected_frames,
                            make_env_fn=make_env_fn, disable_pixel_if_possible=False)

        if cfg.meta and (i + 1) % cfg.meta_test_interval == 0:
            meta_test(cfg, test_make_env_list, test_oracle_context, exploration_policy, world_model, logger,
                      collected_frames)

        if cfg.meta:
            plot_context(cfg, world_model, train_oracle_context, logger, collected_frames)

        logger.dump_scaler(collected_frames)
        exploration_policy.step(current_frames)
        collector.update_policy_weights_()

        if (i + 1) % 10 == 0:
            os.makedirs(os.path.join("checkpoints", str(i)), exist_ok=True)
            torch.save(world_model.state_dict(), os.path.join("checkpoints", str(i), f"world_model.pt"))
            torch.save(actor_model.state_dict(), os.path.join("checkpoints", str(i), f"actor_model.pt"))
            torch.save(value_model.state_dict(), os.path.join("checkpoints", str(i), f"value_model.pt"))

    collector.shutdown()


if __name__ == "__main__":
    main()
