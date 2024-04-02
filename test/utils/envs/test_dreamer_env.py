from functools import partial

import torch
from torchrl.envs import ParallelEnv

from causal_meta.utils.envs.dreamer_env import make_dreamer_env

torch.multiprocessing.set_sharing_strategy("file_system")


def test_make_pomdp_env():
    make_env_fn = partial(make_dreamer_env, env_name="MyCartPole-v0")
    env = make_env_fn()
    td = env.rollout(2, auto_reset=True)

    parallel_env = ParallelEnv(
        num_workers=2,
        create_env_fn=[make_env_fn] * 2,
    )

    td = parallel_env.rollout(2, auto_reset=True)

    parallel_env.close()
