from intact.envs.gym_like.cartpole import CartPoleEnv


def test_heating_env():
    env = CartPoleEnv(render_mode="rgb_array")
    obs, _ = env.reset()
    assert obs is not None

    for i in range(3):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        env.render()
        assert obs is not None
