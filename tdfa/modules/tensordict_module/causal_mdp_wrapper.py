from typing import Any

import torch
from tensordict.nn import TensorDictSequential, TensorDictModule
from tensordict import TensorDict, TensorDictBase

from tdfa.modules.models.causal_world_model import CausalWorldModel


class CausalMDPWrapper(TensorDictModule):
    def __init__(
            self,
            causal_world_model: CausalWorldModel,
    ):
        super().__init__(
            module=causal_world_model,
            in_keys=["observation", "action", "idx"],
            out_keys=["obs_mean", "obs_log_var", "reward", "terminated", "causal_mask"],
        )

    def get_parameter(self, key):
        return self.module.get_parameter(key)

    @property
    def causal_mask(self):
        return self.module.causal_mask

    def parallel_forward(self, tensordict, sampling_times=50):
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


def test_causal_mdp_wrapper():
    obs_dim = 4
    action_dim = 1
    max_context_dim = 0
    task_num = 0
    batch_size = 32

    world_model = CausalWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_context_dim=max_context_dim,
        task_num=task_num,
    )

    td = TensorDict({
        "observation": torch.randn(batch_size, obs_dim),
        "action": torch.randn(batch_size, action_dim),
    },
        batch_size=batch_size,
    )

    causal_mdp_wrapper = CausalMDPWrapper(world_model)

    td = causal_mdp_wrapper(td)

    print(td)


if __name__ == '__main__':
    test_causal_mdp_wrapper()
