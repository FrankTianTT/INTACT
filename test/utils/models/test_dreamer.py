from functools import partial

from intact.utils.envs.dreamer_env import make_dreamer_env
from intact.utils.models.dreamer import make_dreamer
from intact.utils.models.mdp import DreamerConfig


def test_make_causal_dreamer():
    cfg = DreamerConfig()

    make_env_fn = partial(make_dreamer_env, env_name="MyCartPole-v0")
    env = make_env_fn()

    world_model, model_based_env, actor_simulator, value_model, actor_realworld = make_dreamer(cfg, env)
