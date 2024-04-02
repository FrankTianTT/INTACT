import torch
from tensordict import TensorDict
from torchrl.envs import GymEnv, TransformedEnv

from intact.envs.meta_transform import MetaIdxTransform


def test_meta_idx_transform():
    td = TensorDict({"observation": torch.rand(20, 10, 2), "action": torch.rand(20, 10, 2)}, batch_size=(20, 10))
    transform = MetaIdxTransform(0, 10)

    env = GymEnv("Pendulum-v1")
    env = TransformedEnv(env, transform)

    td = env.reset()

    td = env.rollout(10)
