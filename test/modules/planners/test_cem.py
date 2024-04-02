from intact.envs.mdp_env import MDPEnv
from intact.modules.planners.cem import MyCEMPlanner


def test_cem():
    from intact.modules.models.mdp_world_model import CausalWorldModel
    from intact.modules.tensordict_module.mdp_wrapper import MDPWrapper
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

    planner = MyCEMPlanner(
        env=mdp_env,
        planning_horizon=2,
        optim_steps=2,
        num_candidates=3,
        top_k=2,
    )

    td = proof_env.reset()

    mdp_env.reset()
    planner.planning(td)
