import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.model_based import ModelBasedEnvBase

from tdfa.envs import reward_fns_dict, termination_fns_dict


class MyMBEnv(ModelBasedEnvBase):
    def __init__(self, world_model: TensorDictModuleBase, device="cpu", dtype=None, batch_size=None,
                 termination_fns=None, reward_fns=None):
        super().__init__(world_model, device=device, dtype=dtype, batch_size=batch_size)
        self.termination_fns = termination_fns_dict[termination_fns] if termination_fns is not None else None
        self.reward_fns = reward_fns_dict[reward_fns] if reward_fns is not None else None

    def _reset(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        tensordict = TensorDict(
            {},
            batch_size=self.batch_size,
            device=self.device,
        )
        tensordict = tensordict.update(self.state_spec.rand())
        tensordict = tensordict.update(self.observation_spec.rand())
        return tensordict

    def _step(self, tensordict: TensorDict) -> TensorDict:
        tensordict_out = super()._step(tensordict)
        if self.termination_fns is None:
            tensordict_out["terminated"] = tensordict_out["terminated"] > 0  # terminated from world-model are logits
        else:
            tensordict_out["terminated"] = self.termination_fns(
                tensordict["observation"],
                tensordict["action"],
                tensordict_out["observation"]
            )

        tensordict_out["truncated"] = torch.zeros_like(tensordict_out["truncated"]).bool()
        tensordict_out["done"] = torch.logical_or(tensordict_out["terminated"], tensordict_out["truncated"])

        if self.reward_fns is not None:
            tensordict_out["reward"] = self.reward_fns(
                tensordict["observation"],
                tensordict["action"],
                tensordict_out["observation"]
            )
        return tensordict_out

    def set_specs_from_env(self, env: EnvBase):
        # env must be low-dimensional
        super().set_specs_from_env(env)
        self.state_spec = self.observation_spec.clone()
