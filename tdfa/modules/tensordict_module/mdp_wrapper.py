from typing import Any

import torch
from tensordict.nn import TensorDictSequential, TensorDictModule
from tensordict import TensorDict, TensorDictBase

from tdfa.modules.models.mdp_world_model import MDPWorldModel, CausalWorldModel


class MDPWrapper(TensorDictModule):
    def __init__(
            self,
            mdp_world_model: MDPWorldModel,
    ):
        super().__init__(
            module=mdp_world_model,
            in_keys=["observation", "action", "idx"],
            out_keys=["obs_mean", "obs_log_var", "reward", "terminated", "causal_mask"],
        )

        self.causal = isinstance(mdp_world_model, CausalWorldModel)

    def get_parameter(self, key):
        return self.module.get_parameter(key)

    @property
    def causal_mask(self):
        assert self.causal, "causal_mask is only available for CausalWorldModel"
        return self.module.causal_mask

    @property
    def context_model(self):
        return self.module.context_model

    def parallel_forward(self, tensordict, sampling_times=50):
        assert self.causal, "causal_mask is only available for CausalWorldModel"
        assert self.causal_mask.reinforce, "causal_mask should be learned by reinforce"

        assert len(tensordict.batch_size) == 1, "batch_size should be 1-d"
        batch_size = tensordict.batch_size[0]

        repeat_tensordict = tensordict.expand(sampling_times, batch_size).reshape(-1)
        out_tensordict = self.forward(repeat_tensordict, deterministic_mask=False)
        out_tensordict = out_tensordict.reshape(sampling_times, batch_size)

        return out_tensordict

    def forward(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        tensors = tuple(tensordict.get(in_key, None) for in_key in self.in_keys)
        tensors = self._call_module(tensors, **kwargs)
        tensordict_out = self._write_to_tensordict(tensordict, tensors)
        return tensordict_out


def test_plain_mdp_wrapper():
    obs_dim = 4
    action_dim = 1
    batch_size = 32

    world_model = MDPWorldModel(obs_dim=obs_dim, action_dim=action_dim)

    td = TensorDict({
        "observation": torch.randn(batch_size, obs_dim),
        "action": torch.randn(batch_size, action_dim),
    },
        batch_size=batch_size,
    )

    mdp_wrapper = MDPWrapper(world_model)

    td = mdp_wrapper(td)
    assert "obs_mean" in td.keys() and td["obs_mean"].shape == td["observation"].shape, "obs_mean should be in td"
    assert "obs_log_var" in td.keys() and td["obs_log_var"].shape == td["observation"].shape, \
        "obs_log_var should be in td"
    assert "reward" in td.keys() and td["reward"].shape == (batch_size, 1), "reward should be in td"
    assert "terminated" in td.keys() and td["terminated"].shape == (batch_size, 1), "terminated should be in td"


def test_causal_mdp_wrapper():
    obs_dim = 4
    action_dim = 1
    batch_size = 32

    world_model = CausalWorldModel(obs_dim=obs_dim, action_dim=action_dim)

    td = TensorDict({
        "observation": torch.randn(batch_size, obs_dim),
        "action": torch.randn(batch_size, action_dim),
    },
        batch_size=batch_size,
    )

    causal_mdp_wrapper = MDPWrapper(world_model)

    td = causal_mdp_wrapper(td)

    assert "obs_mean" in td.keys() and td["obs_mean"].shape == td["observation"].shape, "obs_mean should be in td"
    assert "obs_log_var" in td.keys() and td["obs_log_var"].shape == td["observation"].shape, \
        "obs_log_var should be in td"
    assert "reward" in td.keys() and td["reward"].shape == (batch_size, 1), "reward should be in td"
    assert "terminated" in td.keys() and td["terminated"].shape == (batch_size, 1), "terminated should be in td"
    assert "causal_mask" in td.keys() and td["causal_mask"].shape == (batch_size, obs_dim + 2, obs_dim + action_dim), \
        "causal_mask should be in td"


if __name__ == '__main__':
    test_plain_mdp_wrapper()
    test_causal_mdp_wrapper()
