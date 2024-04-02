import torch

from causal_meta.modules.models.context_model import ContextModel
from causal_meta.modules.models.policy.actor import Actor


def test_actor_mdp():
    action_dim = 3
    obs_dim = 4
    batch_size = 32
    max_context_dim = 10
    task_num = 100

    context_model = ContextModel(
        meta=True,
        max_context_dim=max_context_dim,
        task_num=task_num,
    )
    actor = Actor(
        state_or_obs_dim=obs_dim,
        action_dim=action_dim,
        context_dim=context_model.max_context_dim,
        is_mdp=True,
    )
    actor.set_context_model(context_model)

    obs = torch.randn(batch_size, obs_dim)
    idx = torch.randint(0, task_num, (batch_size, 1))

    action_mean, action_scale = actor(obs, idx)

    assert action_mean.shape == (batch_size, action_dim)
    assert action_scale.shape == (batch_size, action_dim)


def test_actor_pomdp():
    action_dim = 3
    state_dim = 30
    belief_dim = 200
    batch_size = 32
    max_context_dim = 10
    task_num = 100

    context_model = ContextModel(
        meta=True,
        max_context_dim=max_context_dim,
        task_num=task_num,
    )
    actor = Actor(
        state_or_obs_dim=state_dim,
        belief_dim=belief_dim,
        action_dim=action_dim,
        context_dim=context_model.max_context_dim,
        is_mdp=False,
    )
    actor.set_context_model(context_model)

    state = torch.randn(batch_size, state_dim)
    belief = torch.randn(batch_size, belief_dim)
    idx = torch.randint(0, task_num, (batch_size, 1))

    action_mean, action_scale = actor(state, idx, belief)

    assert action_mean.shape == (batch_size, action_dim)
    assert action_scale.shape == (batch_size, action_dim)
