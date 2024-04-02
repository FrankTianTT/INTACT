from causal_meta.envs.gym_like.cartpole import CartPoleEnv


def test_heating_env():
    env = CartPoleEnv()
    obs, _ = env.reset()
    assert obs is not None

    action = env.action_space.sample()
    obs, _, _, _, _ = env.step(action)
    assert obs is not None
