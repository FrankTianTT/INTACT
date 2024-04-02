import torch
from tensordict import TensorDict

from intact.modules.models.mdp_world_model import PlainMDPWorldModel, CausalWorldModel
from intact.modules.tensordict_module.mdp_wrapper import MDPWrapper


def test_plain_mdp_wrapper():
    obs_dim = 4
    action_dim = 1
    batch_size = 32

    world_model = PlainMDPWorldModel(obs_dim=obs_dim, action_dim=action_dim)

    td = TensorDict(
        {
            "observation": torch.randn(batch_size, obs_dim),
            "action": torch.randn(batch_size, action_dim),
        },
        batch_size=batch_size,
    )

    mdp_wrapper = MDPWrapper(world_model)

    td = mdp_wrapper(td)
    assert "obs_mean" in td.keys() and td["obs_mean"].shape == td["observation"].shape
    assert "obs_log_var" in td.keys() and td["obs_log_var"].shape == td["observation"].shape
    assert "reward_mean" in td.keys() and td["reward_mean"].shape == (batch_size, 1)
    assert "reward_log_var" in td.keys() and td["reward_log_var"].shape == (batch_size, 1)
    assert "terminated" in td.keys() and td["terminated"].shape == (batch_size, 1)


def test_causal_mdp_wrapper():
    obs_dim = 4
    action_dim = 1
    batch_size = 32

    world_model = CausalWorldModel(obs_dim=obs_dim, action_dim=action_dim)

    td = TensorDict(
        {
            "observation": torch.randn(batch_size, obs_dim),
            "action": torch.randn(batch_size, action_dim),
        },
        batch_size=batch_size,
    )

    causal_mdp_wrapper = MDPWrapper(world_model)

    td = causal_mdp_wrapper(td)

    assert "obs_mean" in td.keys() and td["obs_mean"].shape == td["observation"].shape
    assert "obs_log_var" in td.keys() and td["obs_log_var"].shape == td["observation"].shape
    assert "reward_mean" in td.keys() and td["reward_mean"].shape == (batch_size, 1)
    assert "reward_log_var" in td.keys() and td["reward_log_var"].shape == (batch_size, 1)
    assert "terminated" in td.keys() and td["terminated"].shape == (batch_size, 1)
    assert "causal_mask" in td.keys() and td["causal_mask"].shape == (batch_size, obs_dim + 2, obs_dim + action_dim)

    causal_mdp_wrapper.causal_mask
    causal_mdp_wrapper.context_model
    causal_mdp_wrapper.reset()
