from functools import partial

import torch
from tensordict import TensorDict
from omegaconf import DictConfig
from torchrl.envs.transforms import TransformedEnv, Compose, RewardSum, DoubleToFloat, StepCounter
from torchrl.envs.libs import GymEnv, DMControlEnv

from causal_meta.envs.meta_transform import MetaIdxTransform

LIBS = {
    "gym": GymEnv,
    "dm_control": DMControlEnv,
}


def make_mdp_env(env_name, env_kwargs=None, idx=None, task_num=None, pixel=False, env_library="gym"):
    if env_kwargs is None:
        env_kwargs = {}
    if pixel:
        env_kwargs["from_pixels"] = True
        env_kwargs["pixels_only"] = False

    env = LIBS[env_library](env_name, **env_kwargs)

    transforms = [DoubleToFloat(), RewardSum(), StepCounter()]
    if idx is not None:
        transforms.append(MetaIdxTransform(idx, task_num))
    return TransformedEnv(env, transform=Compose(*transforms))
