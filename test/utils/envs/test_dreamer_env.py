from functools import partial

import torch
from torchrl.envs import ParallelEnv

from intact.utils.envs.dreamer_env import make_dreamer_env


def test_make_pomdp_env():
    make_env_fn = partial(make_dreamer_env, env_name="MyCartPole-v0")
    env = make_env_fn()
    td = env.rollout(2, auto_reset=True)
