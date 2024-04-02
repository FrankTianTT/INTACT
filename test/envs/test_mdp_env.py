from causal_meta.envs.mdp_env import MDPEnv


def test_mdp_env():
    from causal_meta.modules.models.mdp_world_model import CausalWorldModel
    from causal_meta.modules.tensordict_module.mdp_wrapper import MDPWrapper
    from torchrl.envs import GymEnv

    obs_dim = 4
    action_dim = 1
    max_context_dim = 0
    task_num = 0

    world_model = CausalWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_context_dim=max_context_dim,
        task_num=task_num,
    )
    causal_mdp_wrapper = MDPWrapper(world_model)

    proof_env = GymEnv("MyCartPole-v0")
    mdp_env = MDPEnv(causal_mdp_wrapper)
    mdp_env.set_specs_from_env(proof_env)

    td = proof_env.reset()

    td = mdp_env.rollout(10, auto_reset=False, tensordict=td, break_when_any_done=False)
    # print(td)
