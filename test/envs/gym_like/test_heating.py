from causal_meta.envs.gym_like.heating import HeatingEnv


def test_heating_env():
    env = HeatingEnv()
    obs, _ = env.reset()
    assert obs is not None

    action = env.action_space.sample()
    obs, _, _, _, _ = env.step(action)
    assert obs is not None
