from itertools import product
from functools import partial
from copy import deepcopy

from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import tensordict
from torchrl.envs import SerialEnv
from torchrl.collectors.collectors import aSyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage

from causal_meta.objectives.causal_dreamer import CausalDreamerModelLoss
from causal_meta.utils.eval import evaluate_policy
from causal_meta.utils.envs import make_dreamer_env
from causal_meta.utils.plot import plot_context


def grad_norm(optimizer: torch.optim.Optimizer):
    sum_of_sq = 0.0
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            if p.grad is not None:
                sum_of_sq += p.grad.pow(2).sum()
    return sum_of_sq.sqrt().detach().item()


def retrieve_stats_from_state_dict(obs_norm_state_dict):
    return {
        "loc": obs_norm_state_dict["loc"],
        "scale": obs_norm_state_dict["scale"],
    }


def train_model(
    cfg,
    replay_buffer,
    world_model,
    world_model_loss,
    model_opt,
    training_steps,
    logits_opt=None,
    logger=None,
    deterministic_mask=False,
    log_prefix="model",
    iters=0,
    only_train=None,
    reward_normalizer=None,
):
    device = next(world_model.parameters()).device
    train_logits_by_reinforce = cfg.model_type == "causal" and cfg.reinforce
    if train_logits_by_reinforce:
        assert logits_opt is not None, "logits_opt should not be None when train logits by reinforce"

    if cfg.model_type == "causal":
        causal_mask = world_model.causal_mask
    else:
        causal_mask = None

    for step in range(training_steps):
        world_model.zero_grad()
        sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(device, non_blocking=True)
        if reward_normalizer:
            reward_normalizer.normalize_reward(sampled_tensordict)

        if reward_normalizer:
            reward_normalizer.normalize_reward(sampled_tensordict)

        if train_logits_by_reinforce and iters % (cfg.train_mask_iters + cfg.train_model_iters) >= cfg.train_model_iters:
            grad, sampling_loss = world_model_loss.reinforce(sampled_tensordict)
            causal_mask = world_model.causal_mask
            logits = causal_mask.mask_logits
            logits.backward(grad)
            logits_opt.step()
        else:
            model_loss_td, sampled_tensordict = world_model_loss(sampled_tensordict)
            total_loss = sum([loss for loss in model_loss_td.values()])

            total_loss.backward()
            clip_grad_norm_(world_model.get_parameter("nets"), cfg.grad_clip)
            model_opt.step()

            logger.add_scaler("world_model/total_loss", total_loss)
            logger.add_scaler("world_model/kl_loss", model_loss_td["loss_model_kl"])
            logger.add_scaler("world_model/reco_loss", model_loss_td["loss_model_reco"])
            logger.add_scaler("world_model/reward_loss", model_loss_td["loss_model_reward"])
            logger.add_scaler("world_model/continue_loss", model_loss_td["loss_model_continue"])

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


def train_agent(
    cfg,
    replay_buffer,
    actor_model,
    actor_loss,
    actor_opt,
    value_model,
    value_loss,
    value_opt,
    training_steps,
    logger=None,
    reward_normalizer=None,
):
    device = next(actor_model.parameters()).device

    for step in range(training_steps):
        sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(device)
        if reward_normalizer:
            reward_normalizer.normalize_reward(sampled_tensordict)

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
        logger.add_scaler("value/mean_continue", (sampled_tensordict[("next", "pred_continue")] > 0).float().mean())
        value_opt.zero_grad()


def reset_module(policy, world_model, new_domain_task_num):
    device = next(policy.parameters()).device
    new_policy = deepcopy(policy).to(device)
    new_world_model = deepcopy(world_model).to(device)

    new_context_model = new_world_model.context_model
    new_context_model.reset(new_domain_task_num)
    new_actor = new_policy.td_module[2][0]
    new_actor.set_context_model(new_context_model)

    return new_policy, new_world_model


def meta_test(cfg, make_env_list, oracle_context, policy, world_model, logger, frames_per_task, adapt_threshold=-3.0):
    logger.dump_scaler(frames_per_task)

    task_num = len(make_env_list)

    policy, world_model = reset_module(policy, world_model, task_num)
    device = next(world_model.parameters()).device

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
    ).to(device)

    collector = aSyncDataCollector(
        create_env_fn=SerialEnv(task_num, make_env_list, shared_memory=False),
        policy=policy,
        total_frames=cfg.meta_task_adjust_frames_per_task,
        frames_per_batch=500,
        init_random_frames=0,
        device=cfg.collector_device,
        storing_device=cfg.collector_device,
        split_trajs=True,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(max_size=cfg.buffer_size),
    )
    world_model_opt = torch.optim.Adam(world_model.get_parameter("context"), lr=cfg.context_lr)

    pbar = tqdm(total=cfg.meta_task_adjust_frames_per_task, desc="meta_test_adjust")
    train_model_iters = 0
    for frame, tensordict in enumerate(collector):
        current_frames = tensordict.get(("collector", "mask")).sum().item()
        pbar.update(current_frames)
        tensordict = match_length(tensordict, cfg.batch_length)
        replay_buffer.extend(tensordict.reshape(-1))

        train_model_iters = train_model(
            cfg,
            replay_buffer,
            world_model,
            world_model_loss,
            world_model_opt,
            cfg.optim_steps_per_batch,
            None,
            logger,
            iters=train_model_iters,
            log_prefix=f"meta_test_model_{frames_per_task}",
        )

        plot_context(cfg, world_model, oracle_context, logger, frame, log_prefix=f"meta_test_model_{frames_per_task}")
        logger.dump_scaler(frame)

    evaluate_policy(
        cfg,
        oracle_context,
        policy,
        logger,
        frames_per_task,
        log_prefix="meta_test",
        make_env_fn=partial(
            make_dreamer_env,
            variable_num=cfg.variable_num,
            state_dim_per_variable=cfg.state_dim_per_variable,
            hidden_dim_per_variable=cfg.belief_dim_per_variable,
        ),
        disable_pixel_if_possible=False,
    )
