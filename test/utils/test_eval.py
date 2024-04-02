from omegaconf import DictConfig
from torchrl.collectors.collectors import RandomPolicy
from torchrl.envs import SerialEnv

from intact.utils.envs.mdp_env import make_mdp_env
from intact.utils.envs.meta_env import create_make_env_list
from intact.utils.eval import evaluate_policy, EvaluateConfig


def test_evaluate_policy():
    eval_cfg = EvaluateConfig()

    env_cfg = DictConfig(
        {
            "meta": True,
            "env_name": "MyCartPole-v0",
            "oracle_context": {
                "gravity": (5.0, 20.0),
            },
            "task_num": 5,
        }
    )
    make_env_list, oracle_context = create_make_env_list(env_cfg, make_mdp_env, mode="meta_train")
    proof_env = SerialEnv(len(make_env_list), make_env_list, shared_memory=False)

    evaluate_policy(eval_cfg, oracle_context, policy=RandomPolicy(proof_env.action_spec))
