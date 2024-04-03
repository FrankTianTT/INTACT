import torch

from intact.modules.models.mdp_world_model.causal_wm import CausalWorldModel


def test_causal_world_model_without_meta():
    obs_dim = 4
    action_dim = 1
    batch_size = 32
    env_num = 5

    world_model = CausalWorldModel(obs_dim=obs_dim, action_dim=action_dim, meta=False)

    for batch_shape in [(), (batch_size,), (env_num, batch_size)]:
        observation = torch.randn(*batch_shape, obs_dim)
        action = torch.randn(*batch_shape, action_dim)

        (
            next_obs_mean,
            next_obs_log_var,
            reward_mean,
            reward_log_var,
            terminated,
            mask,
        ) = world_model(observation, action)

        assert next_obs_mean.shape == next_obs_log_var.shape == (*batch_shape, obs_dim)
        assert reward_mean.shape == terminated.shape == (*batch_shape, 1)
        assert mask.shape == (*batch_shape, obs_dim + 2, obs_dim + action_dim)


def test_causal_world_model_with_meta():
    obs_dim = 4
    action_dim = 1
    max_context_dim = 10
    task_num = 100
    batch_size = 32
    env_num = 5

    world_model = CausalWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        meta=True,
        max_context_dim=max_context_dim,
        task_num=task_num,
    )

    for batch_shape in [(), (batch_size,), (env_num, batch_size)]:
        observation = torch.randn(*batch_shape, obs_dim)
        action = torch.randn(*batch_shape, action_dim)
        idx = torch.randint(0, task_num, (*batch_shape, 1))

        (
            next_obs_mean,
            next_obs_log_var,
            reward_mean,
            reward_log_var,
            terminated,
            mask,
        ) = world_model(observation, action, idx)

        assert next_obs_mean.shape == next_obs_log_var.shape == (*batch_shape, obs_dim)
        assert reward_mean.shape == reward_log_var.shape == terminated.shape == (*batch_shape, 1)
        assert mask.shape == (*batch_shape, obs_dim + 2, obs_dim + action_dim + max_context_dim)


def test_reset():
    obs_dim = 4
    action_dim = 1
    max_context_dim = 10
    task_num = 100

    world_model = CausalWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        meta=True,
        max_context_dim=max_context_dim,
        task_num=task_num,
    )
    world_model.reset()
