from functools import partial

import torch
import gym
from tensordict import TensorDict
from omegaconf import DictConfig
from torchrl.envs.transforms import TransformedEnv, Compose, RewardSum, DoubleToFloat, StepCounter
from torchrl.envs.libs import GymWrapper, DMControlWrapper, GymEnv

from causal_meta.envs.meta_transform import MetaIdxTransform


def make_mdp_env(env_name, env_kwargs=None, idx=None, task_num=None, pixel=False, env_library="gym", max_steps=1000):
    if env_kwargs is None:
        env_kwargs = {}
    if pixel:
        env_kwargs["from_pixels"] = True
        env_kwargs["pixels_only"] = False
    if env_library == "dm_control":
        # env = DMControlEnv(env_name, **env_kwargs)
        raise NotImplementedError
    elif env_library == "gym":
        env = GymEnv(env_name, **env_kwargs, max_episode_steps=max_steps)
        max_steps = None
    else:
        raise ValueError(f"Unknown env library: {env_library}")

    transforms = [DoubleToFloat(), RewardSum(), StepCounter(max_steps)]

    if idx is not None:
        transforms.append(MetaIdxTransform(idx, task_num))
    return TransformedEnv(env, transform=Compose(*transforms))
