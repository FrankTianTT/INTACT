import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torchrl.envs import EnvBase
from torchrl.envs.model_based import ModelBasedEnvBase

from intact.envs import reward_fns_dict, termination_fns_dict


class MDPEnv(ModelBasedEnvBase):
    def __init__(
        self,
        world_model: TensorDictModuleBase,
        device="cpu",
        dtype=None,
        batch_size=None,
        termination_fns="",
        reward_fns="",
    ):
        """
        Args:
            world_model (TensorDictModuleBase): the world model
            device (str, optional): the device to use. Defaults to "cpu".
            dtype (torch.dtype, optional): the data type to use. Defaults to None.
            batch_size (int, optional): the batch size to use. Defaults to None.
            termination_fns (str, optional): the termination function to use. Defaults to "".
            reward_fns (str, optional): the reward function to use. Defaults to "".
        """
        super().__init__(
            world_model, device=device, dtype=dtype, batch_size=batch_size
        )
        self.termination_fns = (
            termination_fns_dict[termination_fns]
            if termination_fns != ""
            else None
        )
        self.reward_fns = (
            reward_fns_dict[reward_fns] if reward_fns != "" else None
        )

    def _reset(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        batch_size = tensordict.batch_size if tensordict is not None else []
        device = tensordict.device if tensordict is not None else self.device
        tensordict = TensorDict(
            {},
            batch_size=batch_size,
            device=device,
        )
        tensordict = tensordict.update(self.state_spec.rand(batch_size))
        tensordict = tensordict.update(self.observation_spec.rand(batch_size))

        return tensordict

    def _step(self, tensordict: TensorDict) -> TensorDict:
        tensordict_out = tensordict.clone()
        tensordict_out = self.world_model(tensordict_out)

        if self.world_model.learn_obs_var:
            obs_std = torch.exp(0.5 * tensordict_out["obs_log_var"])
            reward_std = torch.exp(0.5 * tensordict_out["reward_log_var"])
        else:
            obs_std = torch.zeros_like(tensordict_out["obs_mean"])
            reward_std = torch.zeros_like(tensordict_out["reward_mean"])
        tensordict_out["observation"] = tensordict_out[
            "obs_mean"
        ] + obs_std * torch.randn_like(obs_std)

        if self.termination_fns is None:
            tensordict_out["terminated"] = (
                tensordict_out["terminated"] > 0
            )  # terminated from world-model are logits
        else:
            tensordict_out["terminated"] = self.termination_fns(
                tensordict["observation"],
                tensordict["action"],
                tensordict_out["observation"],
            )

        tensordict_out["truncated"] = torch.zeros_like(
            tensordict_out["truncated"]
        ).bool()
        tensordict_out["done"] = torch.logical_or(
            tensordict_out["terminated"], tensordict_out["truncated"]
        )

        if self.reward_fns is None:
            tensordict_out["reward"] = tensordict_out[
                "reward_mean"
            ] + reward_std * torch.randn_like(reward_std)
        else:
            tensordict_out["reward"] = self.reward_fns(
                tensordict["observation"],
                tensordict["action"],
                tensordict_out["observation"],
            )

        return tensordict_out.select(
            *self.observation_spec.keys(),
            *self.full_done_spec.keys(),
            *self.full_reward_spec.keys(),
            strict=False,
        )

    def set_specs_from_env(self, env: EnvBase):
        # env must be low-dimensional
        super().set_specs_from_env(env)
        self.state_spec = self.observation_spec.clone()
