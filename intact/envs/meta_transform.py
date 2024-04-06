import torch
from tensordict.tensordict import TensorDictBase
from torchrl.data.tensor_specs import (
    TensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
)
from torchrl.envs import Transform


class MetaIdxTransform(Transform):
    def __init__(self, task_idx: int, task_num: int):
        """
        Args:
            task_idx (int): the index of the task
            task_num (int): the number of tasks
        """
        super().__init__(in_keys=None, out_keys=None)
        self.idx = torch.tensor(task_idx).to(torch.long)
        self.task_num = task_num

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict.set("idx", self.idx.expand(*tensordict.batch_size, 1))
        return tensordict

    forward = _call

    def transform_observation_spec(
        self, observation_spec: TensorSpec
    ) -> TensorSpec:
        assert isinstance(observation_spec, CompositeSpec)
        observation_spec["idx"] = DiscreteTensorSpec(
            n=self.task_num,
            shape=torch.Size([1]),
            device=observation_spec.device,
        )
        return observation_spec

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        tensordict_reset.set(
            "idx", self.idx.expand(*tensordict_reset.batch_size, 1)
        )
        return tensordict_reset
