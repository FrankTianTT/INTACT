import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.model_based import ModelBasedEnvBase

from tdfa.envs import reward_fns_dict, termination_fns_dict


class MDPEnv(ModelBasedEnvBase):
    def __init__(
            self, world_model: TensorDictModuleBase,
            device="cpu",
            dtype=None,
            batch_size=None,
            termination_fns=None,
            reward_fns=None
    ):
        super().__init__(world_model, device=device, dtype=dtype, batch_size=batch_size)
        self.termination_fns = termination_fns_dict[termination_fns] if termination_fns is not "" else None
        self.reward_fns = reward_fns_dict[reward_fns] if reward_fns is not "" else None

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
        tensordict_out = tensordict.clone()
        tensordict_out = self.world_model(tensordict_out)

        obs_std = torch.exp(0.5 * tensordict_out["obs_log_var"])
        tensordict_out["observation"] = tensordict_out["obs_mean"] + obs_std * torch.randn_like(obs_std)

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


def test_mdp_env():
    from tdfa.modules.models.mdp_world_model import CausalWorldModel
    from tdfa.modules.tensordict_module.mdp_wrapper import MDPWrapper
    from torchrl.envs import GymEnv

    obs_dim = 4
    action_dim = 1
    max_context_dim = 0
    task_num = 0

    world_model = CausalWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_context_dim=max_context_dim,
        task_num=task_num,
    )
    causal_mdp_wrapper = MDPWrapper(world_model)

    proof_env = GymEnv("CartPoleContinuous-v0")
    mdp_env = MDPEnv(causal_mdp_wrapper)
    mdp_env.set_specs_from_env(proof_env)

    td = proof_env.reset()

    td = mdp_env.rollout(10, auto_reset=False, tensordict=td, break_when_any_done=False)
    print(td)


if __name__ == '__main__':
    test_mdp_env()
