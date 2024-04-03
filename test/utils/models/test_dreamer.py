from functools import partial

from intact.utils.envs.dreamer_env import make_dreamer_env
from intact.utils.models.dreamer import make_dreamer, DreamerConfig


def test_make_causal_dreamer():
    cfg = DreamerConfig()
    env = make_dreamer_env("MyCartPole-v0")
    world_model, model_based_env, actor_simulator, value_model, actor_realworld = make_dreamer(
        cfg, env
    )
