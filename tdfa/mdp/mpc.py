from itertools import product

import tqdm
import hydra
import gym
from gym.envs.mujoco.mujoco_env import BaseMujocoEnv
import torch
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import DoubleToFloat
from torchrl.envs.libs import GymWrapper
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.modules import CEMPlanner, MPPIPlanner
from torchrl.trainers.helpers.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer

from tdfa.mdp.world_model import CausalWorldModel, CausalWorldModelLoss
from tdfa.mdp.model_based_env import MyMBEnv


def get_dim_map(obs_dim, action_dim, context_dim):
    def input_dim_map(dim):
        assert dim < obs_dim + action_dim + context_dim
        if dim < obs_dim:
            return "obs_{}".format(dim)
        elif dim < obs_dim + action_dim:
            return "action_{}".format(dim - obs_dim)
        else:
            return "thata_{}".format(dim - obs_dim - action_dim)

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

    def make_env():
        gym_env = gym.make(cfg.env_name)
        # get BaseMujocoEnv
        sub_env = get_sub_env(gym_env)
        if isinstance(sub_env, BaseMujocoEnv) and sub_env.frame_skip > 1 and cfg.cancel_mujoco_frame_skip:
            original_frame_skip = sub_env.frame_skip
            sub_env.frame_skip = 1
            sub_env.model.opt.timestep *= original_frame_skip

        env = GymWrapper(gym_env)
        return TransformedEnv(env, transform=DoubleToFloat())

    return make_env


def build_world_model(cfg, proof_env):
    obs_dim = proof_env.observation_spec["observation"].shape[0]
    action_dim = proof_env.action_spec.shape[0]

    world_model = CausalWorldModel(obs_dim, action_dim, max_context_dim=cfg.max_context_dim)
    world_model_loss = CausalWorldModelLoss(
        world_model,
        lambda_transition=cfg.lambda_transition,
        lambda_reward=cfg.lambda_reward,
        lambda_terminated=cfg.lambda_terminated,
        sparse_weight=cfg.sparse_weight,
        context_weight=cfg.context_weight,
        sampling_times=cfg.sampling_times,
    )
    model_env = MyMBEnv(
        world_model,
        termination_fns=cfg.termination_fns,
        reward_fns=cfg.reward_fns
                        )
    model_env.set_specs_from_env(proof_env)

    return world_model, world_model_loss, model_env


@hydra.main(version_base="1.1", config_path=".", config_name="config")
def main(cfg):
    if torch.cuda.is_available() and not cfg.model_device != "":
        device = torch.device("cuda:0")
    elif cfg.model_device:
        device = torch.device(cfg.model_device)
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    exp_name = generate_exp_name("MPC", cfg.exp_name)
    logger = get_logger(
        logger_type=cfg.logger,
        logger_name="dreamer",
        experiment_name=exp_name,
        wandb_kwargs={
            "project": "torchrl",
            "group": f"MPC_{cfg.env_name}",
            "offline": cfg.offline_logging,
        },
    )

    make_env = env_constructor(cfg)
    world_model, world_model_loss, model_env = build_world_model(cfg, make_env())

    planner = CEMPlanner(model_env, **cfg.cem_cfg)

    collector = SyncDataCollector(
        create_env_fn=make_env,
        policy=planner,
        total_frames=cfg.total_frames,
        frames_per_batch=cfg.frames_per_batch,
    )

    replay_buffer = TensorDictReplayBuffer()

    init_tensordict = make_env().rollout(cfg.init_random_frames, break_when_any_done=False)
    replay_buffer.extend(init_tensordict)

    module_opt = torch.optim.Adam(world_model.get_parameter("module"), lr=cfg.world_model_lr)
    context_opt = torch.optim.Adam(world_model.get_parameter("context_hat"), lr=0.1)
    mask_opt = torch.optim.Adam(world_model.get_parameter("mask_logits"), lr=cfg.mask_logits_lr)

    input_dim_map, output_dim_map = get_dim_map(4, 1, cfg.max_context_dim)

    pbar = tqdm.tqdm(total=cfg.total_frames)
    collected_frames = 0
    sum_rollout_rewards = 0
    model_learning_steps = 0
    for i, tensordict in enumerate(collector):
        pbar.update(tensordict.numel())

        if tensordict["next", "reward"].shape[1] == 1:
            for j in range(tensordict["next", "reward"].shape[0]):
                sum_rollout_rewards += tensordict["next", "reward"][j].item()
                if tensordict["next", "done"][j].item():
                    logger.log_scalar(
                        "rollout/episode_reward",
                        sum_rollout_rewards,
                        step=collected_frames + j,
                    )
                    sum_rollout_rewards = 0

        current_frames = tensordict.numel()
        collected_frames += current_frames

        replay_buffer.extend(tensordict)
        logger.log_scalar(
            "rollout/step_mean_reward",
            tensordict["next", "reward"].mean().detach().item(),
            step=collected_frames,
        )

        logging_world_model_loss = []
        logging_logits = []
        for j in range(cfg.model_learning_per_step):
            world_model.zero_grad()
            sampled_tensordict = replay_buffer.sample(cfg.batch_size)
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

        if i % cfg.record_interval == 0:
            if len(logging_world_model_loss) > 0:
                loss = torch.stack(logging_world_model_loss).mean(dim=0)
                for dim in range(loss.shape[-1]):
                    logger.log_scalar(
                        "model_loss/{}".format(output_dim_map(dim)),
                        loss[..., dim].detach().mean().item(),
                        step=collected_frames,
                    )
            if len(logging_logits) > 0:
                logits = torch.stack(logging_logits).mean(dim=0)
                for out_dim, in_dim in product(range(logits.shape[0]), range(logits.shape[1])):
                    logger.log_scalar(
                        "mask_logits/{},{}".format(output_dim_map(out_dim), input_dim_map(in_dim)),
                        logits[out_dim, in_dim].detach().item(),
                        step=collected_frames,
                    )

        if i % cfg.test_interval == 0:
            test_env = make_env()  # only for test
            rewards = []
            for j in range(cfg.test_env_nums):
                test_tensordict = test_env.rollout(1000, policy=planner)
                rewards.append(test_tensordict[("next", "reward")].sum())
            logger.log_scalar(
                "test/episode_reward",
                torch.stack(rewards).mean().detach().item(),
                step=collected_frames,
            )


if __name__ == '__main__':
    main()
