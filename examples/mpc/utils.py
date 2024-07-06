from itertools import product
from copy import deepcopy

from tqdm import tqdm
import torch
from tensordict.nn import TensorDictModule, TensorDictModuleWrapper
from torchrl.envs import SerialEnv
from torchrl.trainers.helpers.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import ListStorage

from intact.objectives.mdp.causal_mdp import CausalWorldModelLoss
from intact.envs.mdp_env import MDPEnv
from intact.utils import evaluate_policy, plot_context, match_length


def reset_module(policy, task_num):
    device = next(policy.parameters()).device

    new_policy = deepcopy(policy).to(device)

    sub_module = new_policy
    while not isinstance(sub_module, MDPEnv):
        if isinstance(sub_module, TensorDictModuleWrapper):
            sub_module = sub_module.td_module
        elif isinstance(sub_module, TensorDictModule):
            sub_module = sub_module.module
        else:
            raise ValueError("got {}".format(sub_module))
    new_model_env = sub_module

    new_world_model = new_model_env.world_model
    new_world_model.reset(task_num=task_num)

    return new_policy, new_world_model


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
):
    device = next(world_model.parameters()).device
    train_logits_by_reinforce = cfg.model_type == "causal" and cfg.mask_type == "reinforce" and logits_opt

    if cfg.model_type == "causal":
        causal_mask = world_model.causal_mask
    else:
        causal_mask = None

    for step in range(training_steps):
        world_model.zero_grad()

        sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(device, non_blocking=True)

        if (
            train_logits_by_reinforce
            and iters % (cfg.train_mask_iters + cfg.train_model_iters) >= cfg.train_model_iters
        ):
            grad = world_model_loss.reinforce_forward(sampled_tensordict, only_train)
            causal_mask.mask_logits.backward(grad)
            logits_opt.step()
        else:
            loss_td, total_loss = world_model_loss(sampled_tensordict, deterministic_mask, only_train)
            # context_penalty = (world_model.context_model.context_hat ** 2).sum()
            # total_loss += context_penalty * 0.1
            total_loss.backward()
            model_opt.step()

            if logger is not None:
                for dim in range(loss_td["transition_loss"].shape[-1]):
                    logger.add_scaler(
                        f"{log_prefix}/obs_{dim}",
                        loss_td["transition_loss"][..., dim].mean(),
                    )
                logger.add_scaler(
                    f"{log_prefix}/all_obs_mean",
                    loss_td["transition_loss"].mean(),
                )
                logger.add_scaler(f"{log_prefix}/reward", loss_td["reward_loss"].mean())
                logger.add_scaler(
                    f"{log_prefix}/terminated",
                    loss_td["terminated_loss"].mean(),
                )
                if "mutual_info_loss" in loss_td.keys():
                    logger.add_scaler(
                        f"{log_prefix}/mutual_info_loss",
                        loss_td["mutual_info_loss"].mean(),
                    )
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
                logger.add_scaler(
                    f"{log_prefix}/mask_value({out_name},{in_name})",
                    mask_value[out_dim, in_dim],
                )

        iters += 1
    return iters


def meta_test(
    cfg,
    make_env_list,
    oracle_context,
    policy,
    logger,
    log_idx,
    adapt_threshold=-3.5,
):
    if torch.cuda.is_available():
        device = torch.device(cfg.model_device)
        collector_device = torch.device(cfg.collector_device)
    else:
        device = torch.device("cpu")
        collector_device = torch.device("cpu")

    logger.dump_scaler(log_idx)

    task_num = len(make_env_list)

    policy, world_model = reset_module(policy, task_num=task_num)

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

    collector = SyncDataCollector(
        create_env_fn=SerialEnv(len(make_env_list), make_env_list, shared_memory=False),
        policy=policy,
        total_frames=cfg.meta_test_frames,
        frames_per_batch=cfg.frames_per_batch,
        init_random_frames=0,
        split_trajs=True,
        device=collector_device,
        storing_device=collector_device,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=ListStorage(max_size=cfg.meta_test_frames),
    )
    world_model_opt = torch.optim.Adam(world_model.get_parameter("context"), lr=cfg.context_lr)

    pbar = tqdm(total=cfg.meta_test_frames, desc="meta_test_adjust")
    collected_frames = 0
    for i, tensordict in enumerate(collector):
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
        )
        plot_context(
            cfg,
            world_model,
            oracle_context,
            logger,
            collected_frames,
            log_prefix=f"meta_test_model_{log_idx}",
        )
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
        if world_model.model_type == "causal" and cfg.use_reinforce:
            logits_opt = torch.optim.Adam(
                world_model.get_parameter("context_logits"),
                lr=cfg.context_logits_lr,
            )
        else:
            logits_opt = None

        train_model_iters = 0
        for frame in tqdm(
            range(
                cfg.meta_test_frames,
                3 * cfg.meta_test_frames,
                cfg.frames_per_batch,
            )
        ):
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
            )
            plot_context(
                cfg,
                world_model,
                oracle_context,
                logger,
                frame + cfg.frames_per_batch,
                log_prefix=f"meta_test_model_{log_idx}",
            )
            logger.dump_scaler(frame + cfg.frames_per_batch)
            if cfg.model_type == "causal":
                print("meta test causal mask:")
                print(world_model.causal_mask.printing_mask)

        with torch.no_grad():
            loss_td, all_loss = world_model_loss(sampled_tensordict, deterministic_mask=True)
        mean_transition_loss = loss_td["transition_loss"].mean(0)
        print("mean transition loss after phase (3):", mean_transition_loss)

    evaluate_policy(cfg, oracle_context, policy, logger, log_idx, log_prefix="meta_test")
