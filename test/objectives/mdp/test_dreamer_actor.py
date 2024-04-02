import torch

from intact.modules.models.mdp_world_model import CausalWorldModel
from intact.modules.models.policy.actor import Actor
from intact.modules.models.policy.critic import Critic
from intact.objectives.mdp.dream_actor import DreamActorLoss

#
#
# def test_dreamer_actor():
#     action_dim = 3
#     obs_dim = 4
#     batch_size = 32
#     max_context_dim = 10
#     task_num = 100
#     batch_len = 1
#
#     world_model = CausalWorldModel(
#         obs_dim=obs_dim,
#         action_dim=action_dim,
#         meta=True,
#         max_context_dim=max_context_dim,
#         task_num=task_num,
#     )
#     actor = Actor(
#         state_or_obs_dim=obs_dim,
#         action_dim=action_dim,
#         context_dim=world_model.context_model.max_context_dim,
#         is_mdp=True,
#     )
#     actor.set_context_model(world_model.context_model)
#     critic = Critic(
#         state_or_obs_dim=obs_dim,
#         context_dim=context_model.max_context_dim,
#         is_mdp=True,
#     )
#     critic.set_context_model(context_model)
#
#     loss_model = DreamActorLoss(
#         actor_model=actor,
#     value_model: TensorDictModule,
#     )
#
#     obs = torch.randn(batch_size, obs_dim)
#     idx = torch.randint(0, task_num, (batch_size, 1))
#
#     action_mean, action_scale = actor(obs, idx)
#
#     assert action_mean.shape == (batch_size, action_dim)
#     assert action_scale.shape == (batch_size, action_dim)
