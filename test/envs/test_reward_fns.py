import torch

from intact.envs.reward_fns import heating, ones


def test_heating():
    obs = torch.rand(10, 4) * 3
    act = torch.randn(10, 1)
    next_obs = torch.rand(10, 4) * 3

    heating(obs, act, next_obs)
    ones(obs, act, next_obs)
