from torchrl.envs import Compose

from intact.envs.reward_truncated_transform import RewardTruncatedTransform


def test_reward_truncated_transform():
    from torchrl.envs import GymEnv, TransformedEnv, SerialEnv

    def make_env():
        return GymEnv("CartPole-v1")

    env = SerialEnv(10, make_env)
    env = TransformedEnv(env, Compose(RewardTruncatedTransform()))

    # td = env.reset()

    td = env.rollout(1000, break_when_any_done=False)

    # print(td["next", "reward"].sum(dim=1))
