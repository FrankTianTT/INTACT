from itertools import product

import tqdm
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


def get_sub_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def make_env(env_name):
    gym_env = gym.make(env_name)
    # get BaseMujocoEnv
    sub_env = get_sub_env(gym_env)
    if isinstance(sub_env, BaseMujocoEnv):
        if sub_env.frame_skip > 1:
            original_frame_skip = sub_env.frame_skip
            sub_env.frame_skip = 1
            sub_env.model.opt.timestep *= original_frame_skip

    env = GymWrapper(gym_env)
    return TransformedEnv(env, transform=DoubleToFloat())


def get_dim_map(obs_dim, action_dim, theta_dim):
    def input_dim_map(dim):
        assert dim < obs_dim + action_dim + theta_dim
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


if __name__ == '__main__':
    env_name = "InvertedPendulum-v4"
    exp_name = "default"
    total_frames = 100000
    init_frames = 1000
    train_mask_iters = 10
    train_predictor_iters = 50

    logger = get_logger(
        logger_type="wandb",
        logger_name="mpc",
        experiment_name=generate_exp_name("mpc", exp_name),
        wandb_kwargs={
            "project": "torchrl",
            "group": f"MPC_{env_name}",
            "offline": False,
        },
    )

    proof_env = make_env(env_name)
    world_model = CausalWorldModel(4, 1)
    world_model_loss = CausalWorldModelLoss(world_model)

    fake_env = MyMBEnv(world_model)
    fake_env.set_specs_from_env(proof_env)

    planner = CEMPlanner(
        fake_env,
        planning_horizon=30,
        optim_steps=20,
        num_candidates=11,
        top_k=7
    )

    collector = SyncDataCollector(
        create_env_fn=make_env,
        create_env_kwargs={"env_name": env_name},
        policy=planner,
        total_frames=total_frames,
        frames_per_batch=1,
    )

    replay_buffer = TensorDictReplayBuffer()

    init_tensordict = proof_env.rollout(init_frames, break_when_any_done=False)
    replay_buffer.extend(init_tensordict)

    module_opt = torch.optim.Adam(world_model.get_parameter("module"), lr=0.001)
    theta_opt = torch.optim.Adam(world_model.get_parameter("theta_hat"), lr=0.1)
    mask_opt = torch.optim.Adam(world_model.get_parameter("mask_logits"), lr=0.05)

    input_dim_map, output_dim_map = get_dim_map(4, 1, 0)

    pbar = tqdm.tqdm(total=100000)
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

        for j in range(1):
            world_model.zero_grad()
            sampled_tensordict = replay_buffer.sample(256)
            if model_learning_steps % (train_mask_iters + train_predictor_iters) < train_predictor_iters:
                loss = world_model_loss(sampled_tensordict)
                loss.mean().backward()
                module_opt.step()
                theta_opt.step()

                if model_learning_steps % 5 == 0:
                    for dim in range(loss.shape[-1]):
                        logger.log_scalar(
                            "model_loss/{}".format(output_dim_map(dim)),
                            loss[..., dim].detach().mean().item(),
                            step=collected_frames,
                        )
            else:
                grad = world_model_loss.reinforce(sampled_tensordict)
                logits = world_model.mask_logits
                logits.backward(grad)
                mask_opt.step()

                if model_learning_steps % 5 == 0:
                    for out_dim, in_dim in product(range(logits.shape[0]), range(logits.shape[1])):
                        logger.log_scalar(
                            "mask_logits/{},{}".format(output_dim_map(out_dim), input_dim_map(in_dim)),
                            logits[out_dim, in_dim].detach().item(),
                            step=collected_frames,
                        )
            model_learning_steps += 1

        if collected_frames % 500 == 1:
            rewards = []
            for j in range(5):
                test_tensordict = proof_env.rollout(1000, policy=planner)
                rewards.append(test_tensordict[("next", "reward")].sum())
            logger.log_scalar(
                "test/episode_reward",
                torch.stack(rewards).mean().detach().item(),
                step=collected_frames,
            )
