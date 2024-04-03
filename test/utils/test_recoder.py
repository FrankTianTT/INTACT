from torchrl.envs import GymEnv
from torchrl.collectors.collectors import RandomPolicy

from intact.utils.recoder import Recorder


def test_recoder():
    env = GymEnv("CartPole-v0")
    recoder = Recorder(
        record_interval=1, environment=env, policy_exploration=RandomPolicy(env.action_spec)
    )
    recoder(None)
