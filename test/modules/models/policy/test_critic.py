import torch

from causal_meta.modules.models.context_model import ContextModel
from causal_meta.modules.models.policy.critic import Critic


def test_actor_mdp():
    obs_dim = 4
    batch_size = 32
    max_context_dim = 10
    task_num = 100

    context_model = ContextModel(
        meta=True,
        max_context_dim=max_context_dim,
        task_num=task_num,
    )
    actor = Critic(
        state_or_obs_dim=obs_dim,
        context_dim=context_model.max_context_dim,
        is_mdp=True,
    )
    actor.set_context_model(context_model)

    obs = torch.randn(batch_size, obs_dim)
    idx = torch.randint(0, task_num, (batch_size, 1))

    value = actor(obs, idx)

    assert value.shape == (batch_size, 1)


def test_actor_pomdp():
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
    actor = Critic(
        state_or_obs_dim=state_dim,
        belief_dim=belief_dim,
        context_dim=context_model.max_context_dim,
        is_mdp=False,
    )
    actor.set_context_model(context_model)

    state = torch.randn(batch_size, state_dim)
    belief = torch.randn(batch_size, belief_dim)
    idx = torch.randint(0, task_num, (batch_size, 1))

    value = actor(state, idx, belief)

    assert value.shape == (batch_size, 1)
