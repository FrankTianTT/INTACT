from itertools import product
from functools import partial

from tqdm import tqdm
import hydra
import gym
from gym.envs.mujoco.mujoco_env import BaseMujocoEnv
import torch
from tensordict import TensorDict
from torchrl.envs import TransformedEnv, SerialEnv, DoubleToFloat, Compose, RewardSum
from torchrl.envs.libs import GymWrapper
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.modules import CEMPlanner, MPPIPlanner
from torchrl.trainers.helpers.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer

from tdfa.mdp.world_model import CausalWorldModel, CausalWorldModelLoss
from tdfa.mdp.model_based_env import MyMBEnv
from tdfa.envs.meta_transform import MetaIdxTransform
from tdfa.envs.reward_truncated_transform import RewardTruncatedTransform
from tdfa.utils.metrics import mean_corr_coef


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
    def get_sub_env(env):
        while hasattr(env, "env"):
            env = env.env
        return env

    def make_env(gym_kwargs, idx):
        gym_env = gym.make(cfg.env_name, **gym_kwargs)
        # get BaseMujocoEnv
        sub_env = get_sub_env(gym_env)
        if isinstance(sub_env, BaseMujocoEnv) and sub_env.frame_skip > 1 and cfg.cancel_mujoco_frame_skip:
            original_frame_skip = sub_env.frame_skip
            sub_env.frame_skip = 1
            sub_env.model.opt.timestep *= original_frame_skip

        env = GymWrapper(gym_env)
        transform = Compose(
            DoubleToFloat(),
            MetaIdxTransform(idx, cfg.task_num),
            RewardSum()
        )
        return TransformedEnv(env, transform=transform)

    context_dict = {}
    for key, (low, high) in cfg.oracle_context.items():
        context_dict[key] = torch.rand(cfg.task_num) * (high - low) + low
    context_td = TensorDict(context_dict, batch_size=cfg.task_num)

    make_env_list = []
    for idx in range(cfg.task_num):
        print(idx)
        gym_kwargs = dict([(key, value[idx].item()) for key, value in context_dict.items()])
        make_env_list.append(partial(make_env, gym_kwargs=gym_kwargs, idx=idx))

    return make_env_list, context_td


def build_world_model(cfg, proof_env):
    obs_dim = proof_env.observation_spec["observation"].shape[0]
    action_dim = proof_env.action_spec.shape[0]

    world_model = CausalWorldModel(
        obs_dim,
        action_dim,
        task_num=cfg.task_num,
        max_context_dim=cfg.max_context_dim
    ).to(cfg.device)
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
    ).to(cfg.device)
    model_env = MyMBEnv(
        world_model,
        termination_fns=cfg.termination_fns,
        reward_fns=cfg.reward_fns
    ).to(cfg.device)
    model_env.set_specs_from_env(proof_env)

    return world_model, world_model_loss, model_env, obs_dim, action_dim


@hydra.main(version_base="1.1", config_path=".", config_name="config")
def main(cfg):
    # if torch.cuda.is_available() and not cfg.device != "":
    #     device = torch.device("cuda:0")
    # elif cfg.model_device:
    #     device = torch.device(cfg.model_device)
    # else:
    #     device = torch.device("cpu")
    # print(f"Using device {device}")

    exp_name = generate_exp_name("MPC", cfg.exp_name)
    logger = get_logger(
        logger_type=cfg.logger,
        logger_name="./mpc",
        experiment_name=exp_name,
        wandb_kwargs={
            "project": "torchrl",
            "group": f"MPC_{cfg.env_name}",
            "offline": cfg.offline_logging,
            "config": dict(cfg),
        },
    )

    make_env_list, context_td = env_constructor(cfg)
    assert len(make_env_list) == cfg.task_num > 0
    gt_context = torch.stack(list(context_td.values()), dim=-1)  # only for test
    world_model, world_model_loss, model_env, obs_dim, action_dim = build_world_model(cfg, make_env_list[0]())

    planner = CEMPlanner(
        model_env,
        planning_horizon=cfg.planning_horizon,
        optim_steps=cfg.optim_steps,
        num_candidates=cfg.num_candidates,
        top_k=cfg.top_k,
    )

    collector = SyncDataCollector(
        create_env_fn=SerialEnv(cfg.task_num, make_env_list),
        policy=planner,
        total_frames=cfg.total_frames,
        frames_per_batch=cfg.frames_per_batch,
        init_random_frames=cfg.init_random_frames,
    )

    replay_buffer = TensorDictReplayBuffer()

    module_opt = torch.optim.Adam(world_model.get_parameter("module"), lr=cfg.world_model_lr)
    context_opt = torch.optim.Adam(world_model.get_parameter("context_hat"), lr=cfg.context_lr)
    mask_opt = torch.optim.Adam(world_model.get_parameter("mask_logits"), lr=cfg.mask_logits_lr)

    input_dim_map, output_dim_map = get_dim_map(obs_dim, action_dim, cfg.max_context_dim)

    collected_frames = 0
    model_learning_steps = 0
    for i, tensordict in enumerate(collector):
        current_frames = tensordict.numel()
        collected_frames += current_frames

        def log_scalar(name, value):
            if isinstance(value, torch.Tensor):
                value = value.detach().item()
            logger.log_scalar(name, value, step=collected_frames)

        if tensordict["next", "done"].any():
            episode_reward = tensordict["next", "episode_reward"][tensordict["next", "done"]]
            log_scalar("rollout/episode_reward", episode_reward.mean())
        log_scalar("rollout/step_mean_reward", tensordict["next", "reward"].mean())

        replay_buffer.extend(tensordict.reshape(-1))

        if collected_frames < cfg.init_random_frames:
            continue

        logging_world_model_loss = []
        logging_logits = []
        for j in tqdm(range(cfg.model_learning_per_step * cfg.task_num)):
            world_model.zero_grad()
            sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(cfg.device)
            if model_learning_steps % (cfg.train_mask_iters + cfg.train_predictor_iters) < cfg.train_predictor_iters:
                loss = world_model_loss(sampled_tensordict)
                logging_world_model_loss.append(loss.detach())
                loss.mean().backward()
                module_opt.step()
                context_opt.step()
            else:
                grad = world_model_loss.reinforce(sampled_tensordict)
                logits = world_model.mask_logits
                logging_logits.append(logits.detach())
                logits.backward(grad)
                mask_opt.step()
            model_learning_steps += 1

        if cfg.record_interval < cfg.task_num or i % (cfg.record_interval / cfg.task_num) == 0:
            if len(logging_world_model_loss) > 0:
                loss = torch.stack(logging_world_model_loss).mean(dim=0)
                for dim in range(loss.shape[-1]):
                    if dim >= obs_dim + 2:
                        assert dim == obs_dim + 2  # log mutual info loss
                        log_scalar("model_loss/mutual_info", loss[..., dim].mean())
                    else:
                        log_scalar("model_loss/{}".format(output_dim_map(dim)), loss[..., dim].mean())
            if len(logging_logits) > 0:
                logits = torch.stack(logging_logits).mean(dim=0)
                for out_dim, in_dim in product(range(logits.shape[0]), range(logits.shape[1])):
                    name = "mask_logits/{},{}".format(output_dim_map(out_dim), input_dim_map(in_dim))
                    log_scalar(name, logits[out_dim, in_dim])

                mask = (logits > 0).int()
                valid_context_hat = world_model.context_hat[:, mask.any(dim=0)[-cfg.max_context_dim:]]
                mcc = mean_corr_coef(valid_context_hat.detach().numpy(), gt_context.numpy())
                log_scalar("context/mcc", mcc)

                print("mask_logits:")

                for out_dim in range(logits.shape[0]):
                    for in_dim in range(logits.shape[1]):
                        print(mask[out_dim, in_dim].item(), end=" ")
                    print()

        # if cfg.test_interval < cfg.task_num or i % (cfg.test_interval / cfg.task_num) == 0:
        #     test_env = TransformedEnv(SerialEnv(cfg.task_num, make_env_list),
        #                               RewardTruncatedTransform())  # only for test
        #     rewards = []
        #     for j in range(cfg.test_env_nums):
        #         test_tensordict = test_env.rollout(200, policy=planner, break_when_any_done=False).squeeze()
        #         print(test_tensordict[("next", "reward")].shape)
        #         rewards.extend(test_tensordict[("next", "reward")].sum(dim=-1))
        #     log_scalar(   "test/episode_reward", torch.stack(rewards).mean())

    collector.shutdown()


if __name__ == '__main__':
    main()
