from omegaconf import DictConfig

from causal_meta.utils.envs.mdp_env import make_mdp_env
from causal_meta.utils.envs.meta_env import create_make_env_list


def test_env_constructor():
    cfg = DictConfig(
        {
            "meta": True,
            "env_name": "MyCartPole-v0",
            "oracle_context": {
                "gravity": (5.0, 20.0),
            },
            "task_num": 50,
        }
    )
    make_env_list, oracle_context = create_make_env_list(cfg, make_mdp_env, mode="meta_train")
    env = make_env_list[0]()

    assert oracle_context.shape == (50,)
