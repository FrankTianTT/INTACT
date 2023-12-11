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

from tdfa.helpers.models import make_mlp_model
from tdfa.objectives.causal_mdp import CausalWorldModelLoss
from tdfa.envs.meta_transform import MetaIdxTransform
from tdfa.modules.planners.cem import MyCEMPlanner as CEMPlanner

from matplotlib import pyplot as plt


def get_dim_map(obs_dim, action_dim, context_dim):
    def input_dim_map(dim):
        assert dim < obs_dim + action_dim + context_dim
        if dim < obs_dim:
            return "obs_{}".format(dim)
        elif dim < obs_dim + action_dim:
            return "action_{}".format(dim - obs_dim)
        else:
            return "context_{}".format(dim - obs_dim - action_dim)

    def output_dim_map(dim):
        assert dim < obs_dim + 2
        if dim < obs_dim:
            return "obs_{}".format(dim)
        elif dim < obs_dim + 1:
            return "reward"
        else:
            return "terminated"

    return input_dim_map, output_dim_map


def env_constructor(cfg):
    """ return `make_env` function if `cfg.meta` else return a list of `make_env` function
    """
    if cfg.meta:
        assert cfg.task_num >= 1

    def make_env(gym_kwargs=None, idx=None):
        if gym_kwargs is None:
            gym_kwargs = {}
        env = GymEnv(cfg.env_name, **gym_kwargs)
        transforms = [DoubleToFloat(), RewardSum()]
        if idx is not None:
            transforms.append(MetaIdxTransform(idx, cfg.task_num))
        return TransformedEnv(env, transform=Compose(*transforms))

    if cfg.meta:
        context_dict = {}
        for key, (low, high) in cfg.oracle_context.items():
            context_dict[key] = torch.rand(cfg.task_num) * (high - low) + low
        oracle_context = TensorDict(context_dict, batch_size=cfg.task_num)

        make_env_list = []
        for idx in range(cfg.task_num):
            gym_kwargs = dict([(key, value[idx].item()) for key, value in context_dict.items()])
            make_env_list.append(partial(make_env, gym_kwargs=gym_kwargs, idx=idx))
        return make_env_list, oracle_context
    else:
        return [make_env], None


def test_rollout(env, policy, max_steps, device=torch.device("cpu")):
    tensordicts = []

    tensordict = env.reset().to(device)
    ever_done = torch.zeros(*tensordict.batch_size, 1).to(bool).to(device)
    with torch.no_grad():
        for i in tqdm(range(max_steps)):
            tensordict = policy(tensordict)
            tensordict = env.step(tensordict.cpu()).to(device)
            next_tensordict = step_mdp(tensordict, exclude_action=False)

            tensordict.get(("next", "reward"))[ever_done] = 0
            tensordicts.append(tensordict)

            ever_done |= tensordict.get(("next", "done"))
            if ever_done.all():
                break
            else:
                tensordict = next_tensordict
    batch_size = tensordict.batch_size
    out_td = torch.stack(tensordicts, len(batch_size)).contiguous()
    out_td.refine_names(..., "time")

    return out_td


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

    make_env_list, oracle_context = env_constructor(cfg)
    if oracle_context is not None:
        context_gt = torch.stack(list(oracle_context.values()), dim=-1)  # only for test
    else:
        context_gt = None

    print(make_env_list)

    env_num = len(make_env_list)
    proof_env = make_env_list[0]()
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
        create_env_fn=SerialEnv(len(make_env_list), make_env_list, shared_memory=False),
        policy=explore_policy,
        total_frames=cfg.total_frames,
        frames_per_batch=cfg.frames_per_batch,
        init_random_frames=cfg.init_random_frames,
    )

    replay_buffer = TensorDictReplayBuffer()

    module_opt = torch.optim.Adam(world_model.get_parameter("module"), lr=cfg.world_model_lr)
    context_opt = torch.optim.Adam(world_model.get_parameter("context_hat"), lr=cfg.context_lr)
    if cfg.model_type == "causal":
        observed_logits_opt = torch.optim.Adam(world_model.get_parameter("observed_logits"), lr=cfg.observed_logits_lr)
        context_logits_opt = torch.optim.Adam(world_model.get_parameter("context_logits"), lr=cfg.context_logits_lr)

    input_dim_map, output_dim_map = get_dim_map(
        obs_dim=proof_env.observation_spec["observation"].shape[0],
        action_dim=proof_env.action_spec.shape[0],
        context_dim=cfg.max_context_dim if cfg.meta else 0
    )
    del proof_env

    pbar = tqdm(total=cfg.total_frames)
    collected_frames = 0
    model_learning_steps = 0

    logging_scalar = defaultdict(list)

    def log_scalar(key, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        logging_scalar[key].append(value)

    def train_logits(steps):
        if not (cfg.model_type == "causal" and cfg.reinforce):
            return False
        else:
            return steps % (cfg.train_mask_iters + cfg.train_model_iters) < cfg.train_mask_iters

    for i, tensordict in enumerate(collector):
        pbar.update(tensordict.numel())
        current_frames = tensordict.numel()
        collected_frames += current_frames

        log_scalar("rollout/step_mean_reward", tensordict["next", "reward"].mean())
        if tensordict["next", "done"].any():
            episode_reward = tensordict["next", "episode_reward"][tensordict["next", "done"]]
            log_scalar("rollout/episode_reward", episode_reward.mean())

        replay_buffer.extend(tensordict.reshape(-1))

        if collected_frames < cfg.init_random_frames:
            continue

        for j in range(cfg.model_learning_per_step * env_num):
            world_model.zero_grad()
            sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(device, non_blocking=True)

            if train_logits(model_learning_steps):
                grad = world_model_loss.reinforce(sampled_tensordict)
                logits = world_model.causal_mask.mask_logits

                logits.backward(grad)
                observed_logits_opt.step()
                context_logits_opt.step()
            else:
                loss_td, all_loss = world_model_loss(sampled_tensordict)

                for dim in range(loss_td["transition_loss"].shape[-1]):
                    log_scalar("model_loss/{}".format(f"obs_{dim}"), loss_td["transition_loss"][..., dim].mean())
                log_scalar("model_loss/reward", loss_td["reward_loss"].mean())
                log_scalar("model_loss/terminated", loss_td["terminated_loss"].mean())
                if "mutual_info_loss" in loss_td.keys():
                    log_scalar("model_loss/mutual_info_loss", loss_td["mutual_info_loss"].mean())
                if "context_loss" in loss_td.keys():
                    log_scalar("model_loss/context", loss_td["context_loss"].mean())

                all_loss.backward()
                module_opt.step()
                context_opt.step()

                if cfg.model_type == "causal" and not cfg.reinforce:
                    observed_logits_opt.step()
                    context_logits_opt.step()

            if cfg.model_type == "causal":
                logits = world_model.causal_mask.mask_logits
                for out_dim, in_dim in product(range(logits.shape[0]), range(logits.shape[1])):
                    log_scalar(
                        "mask_logits/{},{}".format(output_dim_map(out_dim), input_dim_map(in_dim)),
                        logits[out_dim, in_dim],
                    )

            model_learning_steps += 1

        if cfg.test_interval < env_num or i % (cfg.test_interval / env_num) == 0:
            test_env = SerialEnv(len(make_env_list), make_env_list, shared_memory=False)
            rewards = []
            for j in range(cfg.test_env_nums):
                test_tensordict = test_rollout(test_env, planner, 200, device)
                rewards.append(test_tensordict[("next", "reward")].sum(dim=1))
            log_scalar("test/episode_reward", torch.stack(rewards).mean())

        if cfg.record_interval < env_num or i % (cfg.record_interval / env_num) == 0:
            context_model = world_model.context_model
            if cfg.meta:
                if cfg.model_type == "causal":
                    valid_context_idx = world_model.causal_mask.valid_context_idx
                else:
                    valid_context_idx = torch.arange(context_model.max_context_dim)
                log_scalar("context/valid_num", len(valid_context_idx))
                mcc, permutation, context_hat = context_model.get_mcc(context_gt, valid_context_idx)
                idxes_gt, idxes_hat = permutation
                log_scalar("context/mcc", mcc)

                if not os.path.exists("context_plot"):
                    os.mkdir("context_plot")

                if len(idxes_gt) == 0:
                    pass
                elif len(idxes_gt) == 1:
                    plt.scatter(context_gt[:, 0], context_hat[:, 0])
                else:
                    num_rows = math.ceil(math.sqrt(len(idxes_gt)))
                    num_cols = math.ceil(len(idxes_gt) / num_rows)
                    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

                    context_names = list(oracle_context.keys())
                    for j, (idx_gt, idx_hat) in enumerate(zip(idxes_gt, idxes_hat)):
                        ax = axs.flatten()[j]
                        ax.set(xlabel=context_names[idx_gt], ylabel="{}th context".format(valid_context_idx[idx_hat]))
                        ax.scatter(context_gt[:, idx_gt], context_hat[:, idx_hat])

                    for j in range(context_gt.shape[1], len(axs.flat)):
                        axs.flat[j].set_visible(False)

                plt.savefig(f"context_plot/{i}.png")
                plt.close()

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
