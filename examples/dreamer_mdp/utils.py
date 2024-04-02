from itertools import product
from copy import deepcopy

from tqdm import tqdm
import torch
from tensordict.nn import TensorDictModule, TensorDictModuleWrapper
from torchrl.envs import SerialEnv
from torchrl.trainers.helpers.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import ListStorage
from torchrl.modules.tensordict_module.exploration import AdditiveGaussianWrapper

from causal_meta.objectives.mdp import CausalWorldModelLoss, DreamActorLoss, DreamCriticLoss
from causal_meta.envs.mdp_env import MDPEnv
from causal_meta.utils import evaluate_policy, plot_context, match_length


def reset_module(world_model, actor, critic, new_domain_task_num):
    device = next(world_model.parameters()).device

    new_actor = deepcopy(actor).to(device)
    new_critic = deepcopy(critic).to(device)
    new_world_model = deepcopy(world_model).to(device)

    new_context_model = new_world_model.context_model
    new_context_model.reset(new_domain_task_num)
    new_actor[0].set_context_model(new_context_model)
    new_critic.set_context_model(new_context_model)

    return new_world_model, new_actor, new_critic


def train_policy(
    cfg,
    replay_buffer,
    actor_loss,
    critic_loss,
    training_steps,
    actor_opt,
    critic_opt,
    logger,
    log_prefix="policy",
    reward_normalizer=None,
):
    device = next(actor_loss.parameters()).device

    for _ in range(training_steps):
        sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(device)
        if reward_normalizer:
            reward_normalizer.normalize_reward(sampled_tensordict)

        actor_loss_td, sampled_tensordict = actor_loss(sampled_tensordict)
        actor_loss_td["loss_actor"].backward()
        actor_opt.step()

        logger.add_scaler(f"{log_prefix}/actor_loss", actor_loss_td["loss_actor"])
        logger.add_scaler(f"{log_prefix}/action_mean", sampled_tensordict["action"].mean())
        logger.add_scaler(f"{log_prefix}/action_std", sampled_tensordict["action"].std())
        logger.add_scaler(f"{log_prefix}/entropy_mean", sampled_tensordict["entropy"].mean())
        actor_opt.zero_grad()

        value_loss_td, sampled_tensordict = critic_loss(sampled_tensordict)
        value_loss_td["loss_value"].backward()
        critic_opt.step()

        logger.add_scaler(f"{log_prefix}/value_loss", value_loss_td["loss_value"])
        logger.add_scaler(f"{log_prefix}/target_mean", sampled_tensordict["lambda_target"].mean())
        logger.add_scaler(f"{log_prefix}/target_std", sampled_tensordict["lambda_target"].std())
        critic_opt.zero_grad()


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
    only_train=None,
    reward_normalizer=None,
):
    device = next(world_model.parameters()).device
    train_logits_by_reinforce = cfg.model_type == "causal" and cfg.reinforce and logits_opt

    if cfg.model_type == "causal":
        causal_mask = world_model.causal_mask
    else:
        causal_mask = None

    for step in range(training_steps):
        world_model.zero_grad()

        sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(device, non_blocking=True)
        if reward_normalizer:
            reward_normalizer.normalize_reward(sampled_tensordict)

        if train_logits_by_reinforce and iters % (cfg.train_mask_iters + cfg.train_model_iters) >= cfg.train_model_iters:
            grad = world_model_loss.reinforce_forward(sampled_tensordict, only_train)
            causal_mask.mask_logits.backward(grad)
            logits_opt.step()
        else:
            loss_td, total_loss = world_model_loss(sampled_tensordict, deterministic_mask, only_train)
            context_penalty = (world_model.context_model.context_hat**2).sum()
            total_loss += context_penalty * 0.5
            total_loss.backward()
            model_opt.step()

            if logger is not None:
                for dim in range(loss_td["transition_loss"].shape[-1]):
                    logger.add_scaler(f"{log_prefix}/obs_{dim}", loss_td["transition_loss"][..., dim].mean())
                logger.add_scaler(f"{log_prefix}/all_obs_mean", loss_td["transition_loss"].mean())
                logger.add_scaler(f"{log_prefix}/reward", loss_td["reward_loss"].mean())
                logger.add_scaler(f"{log_prefix}/terminated", loss_td["terminated_loss"].mean())
                if "mutual_info_loss" in loss_td.keys():
                    logger.add_scaler(f"{log_prefix}/mutual_info_loss", loss_td["mutual_info_loss"].mean())
                if "context_loss" in loss_td.keys():
                    logger.add_scaler(f"{log_prefix}/context", loss_td["context_loss"].mean())

        if cfg.model_type == "causal":
            mask_value = torch.sigmoid(cfg.alpha * causal_mask.mask_logits)
            for out_dim, in_dim in product(range(mask_value.shape[0]), range(mask_value.shape[1])):
                out_name = f"o{out_dim}"
                if in_dim < causal_mask.observed_input_dim:
                    in_name = f"i{in_dim}"
                else:
                    in_name = f"c{in_dim - causal_mask.observed_input_dim}"
                logger.add_scaler(f"{log_prefix}/mask_value({out_name},{in_name})", mask_value[out_dim, in_dim])

        iters += 1
    return iters


def build_loss(cfg, world_model, model_based_env, actor, critic):
    device = next(world_model.parameters()).device

    world_model_loss = CausalWorldModelLoss(
        world_model,
        lambda_transition=cfg.lambda_transition,
        lambda_reward=cfg.lambda_reward if cfg.reward_fns == "" else 0.0,
        lambda_terminated=cfg.lambda_terminated if cfg.termination_fns == "" else 0.0,
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
        lambda_entropy=cfg.lambda_entropy,
    )
    critic_loss = DreamCriticLoss(
        critic,
        discount_loss=cfg.discount_loss,
    )

    return world_model_loss, actor_loss, critic_loss


def meta_test(
    cfg, make_env_list, oracle_context, world_model, actor, critic, logger, log_idx, reward_normalizer, adapt_threshold=-4.0
):
    device = next(world_model.parameters()).device
    logger.dump_scaler(log_idx)
    task_num = len(make_env_list)

    world_model, actor, critic = reset_module(world_model, actor, critic, task_num)
    proof_env = make_env_list[0]()
    policy = AdditiveGaussianWrapper(actor, sigma_init=0.3, sigma_end=0.3, spec=proof_env.action_spec)
    model_based_env = MDPEnv(world_model, termination_fns=cfg.termination_fns, reward_fns=cfg.reward_fns).to(device)
    model_based_env.set_specs_from_env(proof_env)
    del proof_env
    world_model_loss, actor_loss, critic_loss = build_loss(cfg, world_model, model_based_env, actor, critic)

    collector = SyncDataCollector(
        create_env_fn=SerialEnv(len(make_env_list), make_env_list, shared_memory=False),
        policy=policy,
        total_frames=cfg.meta_test_frames,
        frames_per_batch=cfg.frames_per_batch,
        init_random_frames=0,
        split_trajs=True,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=ListStorage(max_size=cfg.meta_test_frames),
    )
    world_model_opt = torch.optim.Adam(world_model.get_parameter("context"), lr=cfg.context_lr)

    pbar = tqdm(total=cfg.meta_test_frames, desc="meta_test_adjust")
    collected_frames = 0
    for i, tensordict in enumerate(collector):
        if reward_normalizer is not None:
            reward_normalizer.update_reward_stats(tensordict)

        current_frames = tensordict.get(("collector", "mask")).sum().item()
        pbar.update(current_frames)
        collected_frames += current_frames
        tensordict = match_length(tensordict, cfg.batch_length)
        tensordict = tensordict.reshape(-1, cfg.batch_length)
        replay_buffer.extend(tensordict)

        train_model(
            cfg,
            replay_buffer,
            world_model,
            world_model_loss,
            training_steps=cfg.optim_steps_per_batch,
            model_opt=world_model_opt,
            logger=logger,
            log_prefix=f"meta_test_model_{log_idx}",
            deterministic_mask=True,
            reward_normalizer=reward_normalizer,
        )
        plot_context(cfg, world_model, oracle_context, logger, collected_frames, log_prefix=f"meta_test_model_{log_idx}")
        logger.dump_scaler(collected_frames)
    pbar.close()
    collector.shutdown()

    if cfg.get("new_oracle_context", None):  # adapt to target domain, only for transition
        with torch.no_grad():
            sampled_tensordict = replay_buffer.sample(len(replay_buffer)).to(device, non_blocking=True)
            loss_td, all_loss = world_model_loss(sampled_tensordict, deterministic_mask=True)
        mean_transition_loss = loss_td["transition_loss"].mean(0)
        adapt_idx = torch.where(mean_transition_loss > adapt_threshold)[0].tolist()
        print("mean transition loss after phase (1):", mean_transition_loss)
        print(adapt_idx)
        if world_model.model_type == "causal":
            world_model.causal_mask.reset(adapt_idx)
            world_model.context_model.fix(world_model.causal_mask.valid_context_idx)

        new_world_model_opt = torch.optim.Adam(world_model.get_parameter("context"), lr=cfg.context_lr)
        new_world_model_opt.add_param_group(dict(params=world_model.get_parameter("nets"), lr=cfg.world_model_lr))
        if world_model.model_type == "causal" and cfg.reinforce:
            logits_opt = torch.optim.Adam(world_model.get_parameter("context_logits"), lr=cfg.context_logits_lr)
        else:
            logits_opt = None
        actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

        train_model_iters = 0
        for frame in tqdm(range(cfg.meta_test_frames, 3 * cfg.meta_test_frames, cfg.frames_per_batch)):
            train_model_iters = train_model(
                cfg,
                replay_buffer,
                world_model,
                world_model_loss,
                training_steps=cfg.optim_steps_per_batch,
                model_opt=new_world_model_opt,
                logits_opt=logits_opt,
                logger=logger,
                log_prefix=f"meta_test_model_{log_idx}",
                iters=train_model_iters,
                only_train=adapt_idx,
                deterministic_mask=False,
                reward_normalizer=reward_normalizer,
            )
            train_policy(
                cfg,
                replay_buffer,
                actor_loss,
                critic_loss,
                cfg.optim_steps_per_batch,
                actor_opt,
                critic_opt,
                logger,
                reward_normalizer=reward_normalizer,
                log_prefix=f"meta_test_policy_{log_idx}",
            )
            plot_context(cfg, world_model, oracle_context, logger, frame, log_prefix=f"meta_test_model_{log_idx}")
            logger.dump_scaler(frame)
            if cfg.model_type == "causal":
                print("envs test causal mask:")
                print(world_model.causal_mask.printing_mask)

    evaluate_policy(cfg, oracle_context, policy, logger, log_idx, log_prefix="meta_test")
