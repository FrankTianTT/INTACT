from intact.envs.gym_like.multi_node import MultiNodeEnv


def test_heating_env():
    env = MultiNodeEnv()
    obs, _ = env.reset()
    assert obs is not None

    action = env.action_space.sample()
    obs, _, _, _, _ = env.step(action)
    assert obs is not None
