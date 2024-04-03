import torch

from intact.modules.models.mdp_world_model import CausalWorldModel
from intact.modules.models.policy.actor import Actor
from intact.modules.models.policy.critic import Critic
from intact.objectives.mdp.dream_actor import DreamActorLoss
from intact.utils.envs.mdp_env import make_mdp_env
from intact.utils.models.mdp import make_mdp_dreamer, MDPConfig


def test_dreamer_actor():
    config = MDPConfig()
    env = make_mdp_env("MyCartPole-v0")
    world_model, model_based_env, actor, critic = make_mdp_dreamer(config, env)

    model_loss = DreamActorLoss(
        actor_model=actor, value_model=critic, model_based_env=model_based_env
    )

    td = env.rollout(10, auto_reset=True)
    td[("collector", "mask")] = torch.ones(10).to(bool)

    model_loss(td)
