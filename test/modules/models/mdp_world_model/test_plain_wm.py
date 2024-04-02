import torch

from intact.modules.models.mdp_world_model.plain_wm import PlainMDPWorldModel


def test_plain_world_model():
    obs_dim = 4
    action_dim = 1
    batch_size = 32
    env_num = 5

    world_model = PlainMDPWorldModel(obs_dim=obs_dim, action_dim=action_dim, meta=False)

    for batch_shape in [(), (batch_size,), (env_num, batch_size)]:
        observation = torch.randn(*batch_shape, obs_dim)
        action = torch.randn(*batch_shape, action_dim)

        next_obs_mean, next_obs_log_var, reward_mean, reward_log_var, terminated = world_model(observation, action)

        assert next_obs_mean.shape == next_obs_log_var.shape == (*batch_shape, obs_dim)
        assert reward_mean.shape == reward_log_var.shape == terminated.shape == (*batch_shape, 1)


def test_reset():
    obs_dim = 4
    action_dim = 1
    task_num = 100
    new_task_num = 10
    max_context_dim = 10
    batch_size = 32

    world_model = PlainMDPWorldModel(
        obs_dim=obs_dim, action_dim=action_dim, meta=True, task_num=task_num, max_context_dim=max_context_dim
    )
    world_model.reset()
