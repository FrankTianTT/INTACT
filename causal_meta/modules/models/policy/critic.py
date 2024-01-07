import torch
from torch import nn
from torchrl.modules.models.models import MLP

from causal_meta.modules.models.context_model import ContextModel


class Critic(nn.Module):
    def __init__(
            self,
            state_or_obs_dim: int,
            context_dim: int,
            is_mdp=True,
            belief_dim=None,
            depth=3,
            num_cells=200,
            activation_class=nn.ELU,
    ):
        self.state_or_obs_dim = state_or_obs_dim
        self.context_dim = context_dim
        self.context_model = None
        self.is_mdp = is_mdp
        self.belief_dim = belief_dim if not is_mdp else 0

        super().__init__()
        self.backbone = MLP(
            in_features=self.state_or_obs_dim + self.belief_dim + self.context_dim,
            out_features=1,
            depth=depth,
            num_cells=num_cells,
            activation_class=activation_class,
        )

    def set_context_model(self, context_model: ContextModel):
        self.context_model = context_model

    def forward(self, state_or_obs, idx, belief=None):
        assert self.context_model is not None, "context model is not set"

        if belief is None:
            inputs = torch.cat([state_or_obs, self.context_model(idx)], dim=-1)
        else:
            inputs = torch.cat([state_or_obs, belief, self.context_model(idx)], dim=-1)
        value = self.backbone(inputs)
        return value


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


if __name__ == '__main__':
    test_actor_mdp()
    test_actor_pomdp()
