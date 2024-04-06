import torch
from tensordict import TensorDict
from torch.optim import Adam

from intact.modules.models.mdp_world_model import CausalWorldModel
from intact.modules.tensordict_module.mdp_wrapper import MDPWrapper
from intact.objectives.mdp.causal_mdp import CausalWorldModelLoss


def test_only_train():
    obs_dim = 4
    action_dim = 1
    max_context_dim = 10
    task_num = 100
    batch_size = 32
    batch_len = 1

    world_model = CausalWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        meta=True,
        max_context_dim=max_context_dim,
        task_num=task_num,
    )
    causal_mdp_wrapper = MDPWrapper(world_model)
    optim = Adam(causal_mdp_wrapper.get_parameter("nets"), lr=1e-3)
    mdp_loss = CausalWorldModelLoss(causal_mdp_wrapper)

    td = TensorDict(
        {
            "observation": torch.randn(batch_size, batch_len, obs_dim),
            "action": torch.randn(batch_size, batch_len, action_dim),
            "idx": torch.randint(0, task_num, (batch_size, batch_len, 1)),
            "next": {
                "terminated": torch.randn(batch_size, batch_len, 1) > 0,
                "reward": torch.randn(batch_size, batch_len, 1),
                "observation": torch.randn(batch_size, batch_len, obs_dim),
            },
            "collector": {
                "mask": torch.ones(batch_size, batch_len, dtype=torch.bool)
            },
        },
        batch_size=(batch_size, batch_len),
    )

    for name, param in world_model.named_parameters():
        if name == "nets.para_mlp.0.weight":
            print("0", param[0, 0])
            print("3", param[3, 0])

    td = causal_mdp_wrapper(td)
    loss_td, total_loss = mdp_loss(td, only_train=[3])
    total_loss.backward()
    optim.step()

    for name, param in world_model.named_parameters():
        if name == "nets.para_mlp.0.weight":
            print("0", param[0, 0])
            print("3", param[3, 0])


def test_reinforce_forward():
    obs_dim = 4
    action_dim = 1
    max_context_dim = 10
    task_num = 100
    batch_size = 32
    batch_len = 1

    world_model = CausalWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        meta=True,
        max_context_dim=max_context_dim,
        task_num=task_num,
    )
    causal_mdp_wrapper = MDPWrapper(world_model)
    optim = Adam(causal_mdp_wrapper.get_parameter("nets"), lr=1e-3)
    mdp_loss = CausalWorldModelLoss(causal_mdp_wrapper)

    td = TensorDict(
        {
            "observation": torch.randn(batch_size, batch_len, obs_dim),
            "action": torch.randn(batch_size, batch_len, action_dim),
            "idx": torch.randint(0, task_num, (batch_size, batch_len, 1)),
            "next": {
                "terminated": torch.randn(batch_size, batch_len, 1) > 0,
                "reward": torch.randn(batch_size, batch_len, 1),
                "observation": torch.randn(batch_size, batch_len, obs_dim),
            },
            "collector": {
                "mask": torch.ones(batch_size, batch_len, dtype=torch.bool)
            },
        },
        batch_size=(batch_size, batch_len),
    )

    td = causal_mdp_wrapper(td)
    mask_grad = mdp_loss.reinforce_forward(td)
