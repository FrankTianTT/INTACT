from functools import partial

import torch

from intact.objectives.causal_dreamer import CausalDreamerModelLoss
from intact.utils.envs.dreamer_env import make_dreamer_env
from intact.utils.models.dreamer import make_dreamer, DreamerConfig


def test_make_causal_dreamer():
    cfg = DreamerConfig()

    make_env_fn = partial(make_dreamer_env, env_name="MyCartPole-v0")
    env = make_env_fn()

    world_model, model_based_env, actor_simulator, value_model, actor_realworld = make_dreamer(
        cfg, env
    )
    model_loss = CausalDreamerModelLoss(world_model=world_model)

    td = env.rollout(10, auto_reset=True)
    td[("collector", "mask")] = torch.ones(10).to(bool)
    model_loss(td)

    td = td.reshape(5, 2)
    model_loss.reinforce_forward(td)
