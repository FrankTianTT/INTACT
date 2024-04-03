from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule

from intact.modules.models.base_world_model import BaseWorldModel
from intact.modules.models.mdp_world_model import PlainMDPWorldModel, CausalWorldModel


class MDPWrapper(TensorDictModule):
    def __init__(
        self,
        mdp_world_model: BaseWorldModel,
    ):
        if isinstance(mdp_world_model, PlainMDPWorldModel):
            out_keys = ["obs_mean", "obs_log_var", "reward_mean", "reward_log_var", "terminated"]
            self.model_type = "plain"
            if isinstance(mdp_world_model, CausalWorldModel):
                out_keys.append("causal_mask")
                self.model_type = "causal"
        # elif isinstance(mdp_world_model, INNWorldModel):
        #     out_keys = ["obs_mean", "obs_log_var", "reward", "terminated", "log_jac_det"]
        #     self.model_type = "inn"
        else:
            raise NotImplementedError

        super().__init__(
            module=mdp_world_model,
            in_keys=["observation", "action", "idx"],
            out_keys=out_keys,
        )

        self.learn_obs_var = mdp_world_model.learn_obs_var

    @property
    def world_model(self) -> BaseWorldModel:
        assert isinstance(self.module, BaseWorldModel)
        return self.module

    def get_parameter(self, key):
        return self.world_model.get_parameter(key)

    @property
    def causal_mask(self):
        assert self.model_type == "causal", "causal_mask is only available for CausalWorldModel"
        return self.world_model.causal_mask

    @property
    def context_model(self):
        return self.world_model.context_model

    def reset(self, task_num=None):
        self.world_model.reset(task_num)

    def parallel_forward(self, tensordict, sampling_times=50):
        assert self.model_type == "causal", "causal_mask is only available for CausalWorldModel"
        assert self.causal_mask.using_reinforce, "causal_mask should be learned by reinforce"

        assert len(tensordict.batch_size) == 1, "batch_size should be 1-d"
        batch_size = tensordict.batch_size[0]

        repeat_tensordict = tensordict.expand(sampling_times, batch_size).reshape(-1)
        out_tensordict = self.forward(repeat_tensordict, deterministic_mask=False)
        out_tensordict = out_tensordict.reshape(sampling_times, batch_size)

        return out_tensordict

    def inv_forward(self, tensordict):
        assert self.model_type == "inn"

        in_keys = [
            "observation",
            "action",
            ("next", "observation"),
            ("next", "reward"),
            ("next", "terminated"),
        ]
        out_keys = ["inv_context", "inv_log_jac_det"]
        tensors = tuple(tensordict.get(in_key, None) for in_key in in_keys)
        tensors = self.world_model.inv_forward(*tensors)

        tensordict_out = self._write_to_tensordict(tensordict, tensors, out_keys=out_keys)
        return tensordict_out

    def forward(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        tensors = tuple(tensordict.get(in_key, None) for in_key in self.in_keys)
        tensors = self.world_model(*tensors, **kwargs)
        tensordict_out = self._write_to_tensordict(tensordict, tensors)
        return tensordict_out
