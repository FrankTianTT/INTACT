import dataclasses
from pathlib import Path

import hydra
import torch
from torch.nn.utils import clip_grad_norm_
import tqdm
from hydra.core.config_store import ConfigStore

from torchrl.envs import EnvBase
from torchrl.modules.tensordict_module.exploration import (
    AdditiveGaussianWrapper,
    OrnsteinUhlenbeckProcessWrapper,
)
from torchrl.objectives.dreamer import (
    DreamerActorLoss,
    DreamerValueLoss,
)
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.trainers.helpers.collectors import (
    make_collector_offpolicy,
    OffPolicyCollectorConfig,
)
from torchrl.trainers.helpers.envs import correct_for_frame_skip
from torchrl.trainers.helpers.logger import LoggerConfig
from torchrl.trainers.helpers.replay_buffer import make_replay_buffer, ReplayArgsConfig
from torchrl.trainers.helpers.trainers import TrainerConfig
from torchrl.trainers.trainers import Recorder, RewardNormalizer

from tdfa.helpers.envs import EnvConfig, make_recorder_env, dreamer_env_constructor, parallel_dreamer_env_constructor
from tdfa.helpers.models import make_causal_dreamer, DreamerConfig
import tdfa.envs
from tdfa.objectives.causal_dreamer import CausalDreamerModelLoss
from dreamer_utils import call_record, grad_norm, match_length, retrieve_stats_from_state_dict

config_fields = [
    (config_field.name, config_field.type, config_field)
    for config_cls in (
        OffPolicyCollectorConfig,
        EnvConfig,
        LoggerConfig,
        ReplayArgsConfig,
        DreamerConfig,
        TrainerConfig,
    )
    for config_field in dataclasses.fields(config_cls)
]
Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base="1.1", config_path=".", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821

    cfg = correct_for_frame_skip(cfg)

    if not isinstance(cfg.reward_scaling, float):
        cfg.reward_scaling = 1.0

    if torch.cuda.is_available() and cfg.model_device == "":
        device = torch.device("cuda:0")
    elif cfg.model_device:
        device = torch.device(cfg.model_device)
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    exp_name = generate_exp_name("Dreamer", cfg.exp_name)
    logger = get_logger(
        logger_type=cfg.logger,
        logger_name="dreamer",
        experiment_name=exp_name,
        wandb_kwargs={
            "project": "torchrl",
            "group": f"Dreamer_{cfg.env_name}",
            "offline": cfg.offline_logging,
            "config": dict(cfg),
        },
    )

    video_tag = f"Dreamer_{cfg.env_name}_policy_test" if cfg.record_video else ""

    obs_norm_state_dict = {"loc": 0.5, "scale": 0.5}

    # Create the different components of dreamer
    world_model, model_based_env, actor_model, value_model, policy = make_causal_dreamer(
        obs_norm_state_dict=obs_norm_state_dict,
        cfg=cfg,
        device=device,
        use_decoder_in_env=True,
        action_key="action",
        value_key="state_value",
        proof_environment=dreamer_env_constructor(
            cfg, stats={"loc": 0.0, "scale": 1.0}
        )(),
    )

    # reward normalization
    if cfg.normalize_rewards_online:
        # if used the running statistics of the rewards are computed and the
        # rewards used for training will be normalized based on these.
        reward_normalizer = RewardNormalizer(
            scale=cfg.normalize_rewards_online_scale,
            decay=cfg.normalize_rewards_online_decay,
        )
    else:
        reward_normalizer = None

    # Losses
    world_model_loss = CausalDreamerModelLoss(
        world_model,
        lambda_kl=cfg.lambda_kl,
        lambda_reco=cfg.lambda_reco,
        lambda_reward=cfg.lambda_reward,
        lambda_continue=cfg.lambda_continue,
    )
    actor_loss = DreamerActorLoss(
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

    # Exploration noise to be added to the actions
    if cfg.exploration == "additive_gaussian":
        exploration_policy = AdditiveGaussianWrapper(
            policy,
            # sigma_init=0.3,
            # sigma_end=0.3,
            annealing_num_steps=cfg.total_frames,
        ).to(device)
    elif cfg.exploration == "ou_exploration":
        exploration_policy = OrnsteinUhlenbeckProcessWrapper(
            policy,
            annealing_num_steps=cfg.total_frames,
        ).to(device)
    elif cfg.exploration == "":
        exploration_policy = policy.to(device)

    action_dim_gsde, state_dim_gsde = None, None
    create_env_fn = parallel_dreamer_env_constructor(
        cfg=cfg,
        obs_norm_state_dict=obs_norm_state_dict,
        action_dim_gsde=action_dim_gsde,
        state_dim_gsde=state_dim_gsde,
    )
    if isinstance(create_env_fn, EnvBase):
        create_env_fn.rollout(2)
    else:
        create_env_fn().rollout(2)

    # Create the replay buffer

    collector = make_collector_offpolicy(
        make_env=create_env_fn,
        actor_model_explore=exploration_policy,
        cfg=cfg,
    )
    print("collector:", collector)

    replay_buffer = make_replay_buffer("cpu", cfg)

    record = Recorder(
        record_frames=cfg.record_frames,
        frame_skip=cfg.frame_skip,
        policy_exploration=policy,
        environment=make_recorder_env(
            cfg=cfg,
            video_tag=video_tag,
            obs_norm_state_dict=obs_norm_state_dict,
            logger=logger,
            create_env_fn=create_env_fn,
        ),
        record_interval=cfg.record_interval,
        log_keys=cfg.recorder_log_keys,
    )

    final_seed = collector.set_seed(cfg.seed)
    print(f"init seed: {cfg.seed}, final seed: {final_seed}")
    # Training loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.total_frames)
    path = Path("./log")
    path.mkdir(exist_ok=True)

    # optimizers
    world_model_opt = torch.optim.Adam(world_model.get_parameter("module"), lr=cfg.world_model_lr)
    context_opt = torch.optim.Adam(world_model.get_parameter("context"), lr=cfg.context_lr)
    mask_logits_opt = torch.optim.Adam(world_model.get_parameter("mask_logits"), lr=cfg.mask_logits_lr)
    actor_opt = torch.optim.Adam(actor_model.parameters(), lr=cfg.actor_value_lr)
    value_opt = torch.optim.Adam(value_model.parameters(), lr=cfg.actor_value_lr)

    for i, tensordict in enumerate(collector):
        cmpt = 0
        if reward_normalizer is not None:
            reward_normalizer.update_reward_stats(tensordict)
        current_frames = tensordict.get(("collector", "mask")).sum().item()

        pbar.update(current_frames)
        collected_frames += current_frames

        def log_scalar(key, value):
            if isinstance(value, torch.Tensor):
                value = value.detach().item()
            logger.log_scalar(key, value, step=collected_frames)

        tensordict = match_length(tensordict, cfg.batch_length)
        tensordict = tensordict.reshape(-1, cfg.batch_length)
        replay_buffer.extend(tensordict.cpu())

        mask = tensordict.get(("collector", "mask"))
        episode_reward = tensordict.get(("next", "episode_reward"))[mask]
        done = tensordict.get(("next", "done"))[mask]
        mean_episode_reward = episode_reward[done].mean()
        log_scalar("rollout/reward_mean", tensordict[("next", "reward")][mask].mean())
        log_scalar("rollout/reward_std", tensordict[("next", "reward")][mask].std())
        log_scalar("rollout/episode_reward_mean", mean_episode_reward)
        log_scalar("rollout/action_mean", tensordict["action"][mask].mean())
        log_scalar("rollout/action_std", tensordict["action"][mask].std())


        if (i % cfg.record_interval) == 0:
            do_log = True
        else:
            do_log = False

        if collected_frames >= cfg.init_random_frames:
            if i % cfg.record_interval == 0:
                log_scalar("cmpt", cmpt)
            for j in range(cfg.optim_steps_per_batch):
                cmpt += 1
                # sample from replay buffer
                sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(
                    device, non_blocking=True
                )
                if reward_normalizer is not None:
                    sampled_tensordict = reward_normalizer.normalize_reward(
                        sampled_tensordict
                    )
                # update world model
                model_loss_td, sampled_tensordict = world_model_loss(
                    sampled_tensordict
                )
                loss_world_model = (
                        model_loss_td["loss_model_kl"]
                        + model_loss_td["loss_model_reco"]
                        + model_loss_td["loss_model_reward"]
                        + model_loss_td["loss_model_continue"]
                )
                # If we are logging videos, we keep some frames.
                if (
                        cfg.record_video
                        and (record._count + 1) % cfg.record_interval == 0
                ):
                    sampled_tensordict_save = (
                        sampled_tensordict.select(
                            "next" "state",
                            "belief",
                        )[:4]
                        .detach()
                        .to_tensordict()
                    )
                else:
                    sampled_tensordict_save = None

                loss_world_model.backward()
                clip_grad_norm_(world_model.get_parameter("module"), cfg.grad_clip)
                world_model_opt.step()

                if j == cfg.optim_steps_per_batch - 1 and do_log:
                    log_scalar("world_model/total_loss", loss_world_model)
                    log_scalar("world_model/grad", grad_norm(world_model_opt))
                    log_scalar("world_model/kl_loss", model_loss_td["loss_model_kl"])
                    log_scalar("world_model/reco_loss", model_loss_td["loss_model_reco"])
                    log_scalar("world_model/reward_loss", model_loss_td["loss_model_reward"])
                    log_scalar("world_model/continue_loss", model_loss_td["loss_model_continue"])
                world_model_opt.zero_grad()

                if collected_frames >= cfg.train_agent_frames:
                    # update actor network
                    actor_loss_td, sampled_tensordict = actor_loss(sampled_tensordict)
                    actor_loss_td["loss_actor"].backward()
                    clip_grad_norm_(actor_model.parameters(), cfg.grad_clip)
                    actor_opt.step()
                    if j == cfg.optim_steps_per_batch - 1 and do_log:
                        log_scalar("actor/loss", actor_loss_td["loss_actor"])
                        log_scalar("actor/grad", grad_norm(actor_opt))
                        log_scalar("actor/action_mean", sampled_tensordict["action"].mean())
                        log_scalar("actor/action_std", sampled_tensordict["action"].std())
                    actor_opt.zero_grad()

                    # update value network
                    value_loss_td, sampled_tensordict = value_loss(sampled_tensordict)
                    value_loss_td["loss_value"].backward()
                    clip_grad_norm_(value_model.parameters(), cfg.grad_clip)
                    value_opt.step()
                    if j == cfg.optim_steps_per_batch - 1 and do_log:
                        log_scalar("value/loss", value_loss_td["loss_value"])
                        log_scalar("value/grad", grad_norm(value_opt))
                        log_scalar("value/target_mean", sampled_tensordict["lambda_target"].mean())
                        log_scalar("value/target_std", sampled_tensordict["lambda_target"].std())
                    value_opt.zero_grad()
                    if j == cfg.optim_steps_per_batch - 1:
                        do_log = False

            stats = retrieve_stats_from_state_dict(obs_norm_state_dict)
            call_record(
                logger,
                record,
                collected_frames,
                None,
                stats,
                model_based_env,
                actor_model,
                cfg,
            )
        if cfg.exploration != "":
            exploration_policy.step(current_frames)
        collector.update_policy_weights_()


if __name__ == "__main__":
    main()
