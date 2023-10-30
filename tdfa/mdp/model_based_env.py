import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.model_based import ModelBasedEnvBase


class MyMBEnv(ModelBasedEnvBase):
    def __init__(self, world_model, device="cpu", dtype=None, batch_size=None):
        super().__init__(world_model, device=device, dtype=dtype, batch_size=batch_size)

    def _reset(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        tensordict = TensorDict(
            {},
            batch_size=self.batch_size,
            device=self.device,
        )
        tensordict = tensordict.update(self.state_spec.rand())
        tensordict = tensordict.update(self.observation_spec.rand())
        return tensordict

    def _step(self, tensordict: TensorDict, ) -> TensorDict:
        tensordict_out = super()._step(tensordict)
        tensordict_out["terminated"] = tensordict_out["terminated"] > 0  # terminated from world-model are logits
        tensordict_out["truncated"] = torch.zeros_like(tensordict_out["truncated"]).bool()
        tensordict_out["done"] = torch.logical_or(tensordict_out["terminated"], tensordict_out["truncated"])
        return tensordict_out

    def set_specs_from_env(self, env: EnvBase):
        # env must be low-dimensional
        super().set_specs_from_env(env)
        self.state_spec = self.observation_spec.clone()
