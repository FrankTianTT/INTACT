from functools import partial
from itertools import product
import os

import hydra
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tensordict.nn.probabilistic import InteractionType
from torchrl.envs import ParallelEnv, SerialEnv, TransformedEnv
from torchrl.record import VideoRecorder
from torchrl.modules.tensordict_module.exploration import AdditiveGaussianWrapper
from torchrl.objectives.dreamer import DreamerActorLoss, DreamerValueLoss
from torchrl.trainers.helpers.collectors import SyncDataCollector, MultiaSyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.trainers.trainers import Recorder, RewardNormalizer

from causal_meta.helpers.envs import make_dreamer_env, create_make_env_list
from causal_meta.helpers.models import make_causal_dreamer
from causal_meta.helpers.logger import build_logger
from causal_meta.objectives.causal_dreamer import CausalDreamerModelLoss
from causal_meta.helpers.reocoder import Recorder

from utils import grad_norm, match_length


@hydra.main(version_base="1.1", config_path="conf", config_name="main")
def main(cfg: "DictConfig"):  # noqa: F821
    if torch.cuda.is_available():
        device = torch.device(cfg.model_device)
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

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
    world_model, model_based_env, actor_model, value_model, policy = make_causal_dreamer(
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

    collector = MultiaSyncDataCollector(
        # create_env_fn=SerialEnv(task_num, train_make_env_list),
        create_env_fn=train_make_env_list,
        policy=exploration_policy,
        total_frames=cfg.train_frames_per_task * task_num,
        frames_per_batch=cfg.frames_per_batch,
        init_random_frames=cfg.init_frames_per_task * task_num,
        device=cfg.collector_device,
        storing_device=cfg.collector_device,
        split_trajs=True
    )

    # eval_env = SerialEnv(task_num, train_make_env_list)
    # eval_env = TransformedEnv(eval_env, VideoRecorder(logger, "eval"))
    record = Recorder(
        env_max_steps=cfg.env_max_steps,
        eval_repeat_times=cfg.eval_repeat_times,
        policy_exploration=policy,
        environment=train_make_env_list[0](),
        record_interval=cfg.record_interval,
        # exploration_type=InteractionType.MODE
        exploration_type=InteractionType.RANDOM
    )

    # replay buffer
    buffer_size = cfg.train_frames_per_task * task_num if cfg.buffer_size == -1 else cfg.buffer_size
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(max_size=buffer_size),
    )
    final_seed = collector.set_seed(cfg.seed)
    print(f"init seed: {cfg.seed}, final seed: {final_seed}")

    # optimizers
    world_model_opt = torch.optim.Adam(world_model.get_parameter("nets"), lr=cfg.world_model_lr)
    world_model_opt.add_param_group(dict(params=world_model.get_parameter("context"), lr=cfg.context_lr))
    if cfg.model_type == "causal":
        logits_opt = torch.optim.Adam(world_model.get_parameter("observed_logits"), lr=cfg.observed_logits_lr)
        logits_opt.add_param_group(dict(params=world_model.get_parameter("context_logits"), lr=cfg.context_logits_lr))
    actor_opt = torch.optim.Adam(actor_model.parameters(), lr=cfg.actor_value_lr)
    value_opt = torch.optim.Adam(value_model.parameters(), lr=cfg.actor_value_lr)

    # Training loop
    collected_frames = 0
    pbar = tqdm(total=cfg.train_frames_per_task * task_num)
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

        if collected_frames < cfg.init_frames_per_task * task_num:
            continue

        for j in range(cfg.optim_steps_per_batch):
            world_model.zero_grad()
            sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(device, non_blocking=True)
            if reward_normalizer is not None:
                sampled_tensordict = reward_normalizer.normalize_reward(sampled_tensordict)

            if cfg.model_type == "causal" and j % (
                    cfg.train_mask_iters + cfg.train_model_iters) >= cfg.train_model_iters:
                grad, sampling_loss = world_model_loss.reinforce(sampled_tensordict)
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
                    logger.add_scaler(f"causal/logits({out_name},{in_name})", logits[out_dim, in_dim])
                for out_dim in range(logits.shape[0]):
                    logger.add_scaler(f"causal/sampling_loss_{out_dim}", sampling_loss[..., out_dim].mean())
            else:
                model_loss_td, sampled_tensordict = world_model_loss(
                    sampled_tensordict
                )
                loss_world_model = (
                        model_loss_td["loss_model_kl"]
                        + model_loss_td["loss_model_reco"]
                        + model_loss_td["loss_model_reward"]
                        + model_loss_td["loss_model_continue"]
                )

                loss_world_model.backward()
                clip_grad_norm_(world_model.get_parameter("nets"), cfg.grad_clip)
                world_model_opt.step()

                logger.add_scaler("world_model/total_loss", loss_world_model)
                logger.add_scaler("world_model/grad", grad_norm(world_model_opt))
                logger.add_scaler("world_model/kl_loss", model_loss_td["loss_model_kl"])
                logger.add_scaler("world_model/reco_loss", model_loss_td["loss_model_reco"])
                logger.add_scaler("world_model/reward_loss", model_loss_td["loss_model_reward"])
                logger.add_scaler("world_model/continue_loss", model_loss_td["loss_model_continue"])
                collector_mask = sampled_tensordict.get(("collector", "mask"))
                prior_mean = sampled_tensordict[("next", "prior_mean")][collector_mask]
                prior_std = sampled_tensordict[("next", "prior_std")][collector_mask]
                posterior_mean = sampled_tensordict[("next", "posterior_mean")][collector_mask]
                posterior_std = sampled_tensordict[("next", "posterior_std")][collector_mask]
                # print(prior_mean[0], prior_std[0], posterior_mean[0], posterior_std[0])
                logger.add_scaler("world_model/mean_prior_mean", prior_mean.mean())
                logger.add_scaler("world_model/mean_prior_std", prior_std.mean())
                logger.add_scaler("world_model/mean_posterior_mean", posterior_mean.mean())
                logger.add_scaler("world_model/mean_posterior_std", posterior_std.mean())
                logger.add_scaler("world_model/mean_std_ratio", torch.log(prior_std / posterior_std).mean())
                logger.add_scaler("world_model/mean_mean_diff", torch.pow(prior_mean - posterior_mean, 2).mean())

                # mean_continue = (sampled_tensordict[("next", "pred_continue")] > 0).float().mean()
                # logger.add_scaler("world_model/mean_continue", mean_continue)

            if collected_frames < cfg.policy_learning_frames_per_task * task_num:
                continue
            # update policy network
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

        record.suffix = f"_{i}"
        td_record = record(None)
        if td_record is not None:
            if "r_evaluation" in td_record.keys():
                logger.add_scaler("eval/reward", td_record["r_evaluation"])
            if "total_r_evaluation" in td_record.keys():
                logger.add_scaler("eval/episode", td_record["total_r_evaluation"])
        if cfg.model_type == "causal":
            print()
            print(world_model.causal_mask.printing_mask)

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
