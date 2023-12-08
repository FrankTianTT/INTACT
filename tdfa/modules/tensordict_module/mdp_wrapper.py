from typing import Any

import torch
from tensordict.nn import TensorDictSequential, TensorDictModule
from tensordict import TensorDict, TensorDictBase

from tdfa.modules.models.mdp_world_model import BaseMDPWorldModel, PlainMDPWorldModel, CausalWorldModel, INNWorldModel


class MDPWrapper(TensorDictModule):
    def __init__(
            self,
            mdp_world_model: BaseMDPWorldModel,
    ):
        if isinstance(mdp_world_model, PlainMDPWorldModel):
            out_keys = ["obs_mean", "obs_log_var", "reward", "terminated"]
            self.model_type = "plain"
            if isinstance(mdp_world_model, CausalWorldModel):
                out_keys.append("causal_mask")
                self.model_type = "causal"
        elif isinstance(mdp_world_model, INNWorldModel):
            out_keys = ["obs_mean", "obs_log_var", "reward", "terminated", "log_jac_det"]
            self.model_type = "inn"
        else:
            raise NotImplementedError

        super().__init__(
            module=mdp_world_model,
            in_keys=["observation", "action", "idx"],
            out_keys=out_keys,
        )

        self.causal = isinstance(mdp_world_model, CausalWorldModel)
        self.learn_obs_var = mdp_world_model.learn_obs_var

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
        assert self.model_type == "causal", "causal_mask is only available for CausalWorldModel"
        assert self.causal_mask.reinforce, "causal_mask should be learned by reinforce"

        assert len(tensordict.batch_size) == 1, "batch_size should be 1-d"
        batch_size = tensordict.batch_size[0]

        repeat_tensordict = tensordict.expand(sampling_times, batch_size).reshape(-1)
        out_tensordict = self.forward(repeat_tensordict, deterministic_mask=False)
        out_tensordict = out_tensordict.reshape(sampling_times, batch_size)

        return out_tensordict

    def inv_forward(self, tensordict):
        assert self.model_type == "inn"

        in_keys = ["observation", "action", ("next", "observation"), ("next", "reward"), ("next", "terminated")]
        out_keys = ["inv_context", "inv_log_jac_det"]
        tensors = tuple(tensordict.get(in_key, None) for in_key in in_keys)
        tensors = self.module.inv_forward(*tensors)

        tensordict_out = self._write_to_tensordict(tensordict, tensors, out_keys=out_keys)
        return tensordict_out

    def forward(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        tensors = tuple(tensordict.get(in_key, None) for in_key in self.in_keys)
        tensors = self.module(*tensors, **kwargs)
        tensordict_out = self._write_to_tensordict(tensordict, tensors)
        return tensordict_out


def test_plain_mdp_wrapper():
    obs_dim = 4
    action_dim = 1
    batch_size = 32

    world_model = PlainMDPWorldModel(obs_dim=obs_dim, action_dim=action_dim)

    td = TensorDict({
        "observation": torch.randn(batch_size, obs_dim),
        "action": torch.randn(batch_size, action_dim),
    },
        batch_size=batch_size,
    )

    mdp_wrapper = MDPWrapper(world_model)

    td = mdp_wrapper(td)
    assert "obs_mean" in td.keys() and td["obs_mean"].shape == td["observation"].shape
    assert "obs_log_var" in td.keys() and td["obs_log_var"].shape == td["observation"].shape
    assert "reward" in td.keys() and td["reward"].shape == (batch_size, 1)
    assert "terminated" in td.keys() and td["terminated"].shape == (batch_size, 1)


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

    assert "obs_mean" in td.keys() and td["obs_mean"].shape == td["observation"].shape
    assert "obs_log_var" in td.keys() and td["obs_log_var"].shape == td["observation"].shape
    assert "reward" in td.keys() and td["reward"].shape == (batch_size, 1)
    assert "terminated" in td.keys() and td["terminated"].shape == (batch_size, 1)
    assert "causal_mask" in td.keys() and td["causal_mask"].shape == (batch_size, obs_dim + 2, obs_dim + action_dim)


def test_inn_mdp_wrapper():
    obs_dim = 4
    action_dim = 1
    batch_size = 32
    task_num = 100

    world_model = INNWorldModel(obs_dim=obs_dim, action_dim=action_dim, task_num=task_num)

    td = TensorDict({
        "observation": torch.randn(batch_size, obs_dim),
        "action": torch.randn(batch_size, action_dim),
        "idx": torch.randint(0, task_num, (batch_size, 1)),
        "next": {
            "observation": torch.randn(batch_size, obs_dim),
            "reward": torch.randn(batch_size, 1),
            "terminated": torch.randn(batch_size, 1),
        }
    },
        batch_size=batch_size,
    )

    inn_mdp_wrapper = MDPWrapper(world_model)

    td = inn_mdp_wrapper(td)

    assert "obs_mean" in td.keys() and td["obs_mean"].shape == td["observation"].shape
    assert "obs_log_var" in td.keys() and td["obs_log_var"].shape == td["observation"].shape
    assert "reward" in td.keys() and td["reward"].shape == (batch_size, 1)
    assert "terminated" in td.keys() and td["terminated"].shape == (batch_size, 1)
    assert "log_jac_det" in td.keys() and td["log_jac_det"].shape == (batch_size, 1)

    td = inn_mdp_wrapper.inv_forward(td)

    assert "inv_context" in td.keys() and td["inv_context"].shape == (batch_size, obs_dim + 2)
    assert "inv_log_jac_det" in td.keys() and td["inv_log_jac_det"].shape == (batch_size, 1)


if __name__ == '__main__':
    # test_plain_mdp_wrapper()
    # test_causal_mdp_wrapper()
    test_inn_mdp_wrapper()
