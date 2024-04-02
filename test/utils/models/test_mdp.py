from intact.utils.envs.mdp_env import make_mdp_env
from intact.utils.models.mdp import make_mdp_model, MDPConfig


def test_make_mdp_model():
    config = MDPConfig()
    env = make_mdp_env("MyCartPole-v0")
    world_model, model_based_env = make_mdp_model(config, env)
