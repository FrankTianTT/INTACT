from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Tuple, Union

import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import expand_as_right, NestedKey
from torchrl.envs import Transform
from torchrl.data.tensor_specs import TensorSpec, CompositeSpec, DiscreteTensorSpec


class MetaIdxTransform(Transform):
    def __init__(
            self,
            task_idx: int,
            task_num: int
    ):
        super().__init__(in_keys=None, out_keys=None)
        self.idx = torch.tensor(task_idx).to(torch.long)
        self.task_num = task_num

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self(tensordict)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict.set("idx", self.idx.expand(*tensordict.batch_size, 1))
        return tensordict

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        assert isinstance(observation_spec, CompositeSpec)
        observation_spec["idx"] = DiscreteTensorSpec(
            n=self.task_num,
            shape=torch.Size([1]),
            device=observation_spec.device
        )
        return observation_spec

    def _reset(
            self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        tensordict_reset.set("idx", self.idx.expand(*tensordict_reset.batch_size, 1))
        return tensordict_reset


if __name__ == '__main__':
    td = TensorDict(
        {
            "observation": torch.rand(20, 10, 2),
            "action": torch.rand(20, 10, 2)
        },
        batch_size=(20, 10)
    )
    transform = MetaIdxTransform(0, 10)

    # print(transform.in_keys, transform.out_keys)

    # print(transform(td))

    from torchrl.envs import GymEnv, TransformedEnv

    env = GymEnv("Pendulum-v1")
    env = TransformedEnv(env, transform)

    td = env.reset()
    print(td)

    td = env.rollout(10)
    print(td)
    print(td.get("idx"))
