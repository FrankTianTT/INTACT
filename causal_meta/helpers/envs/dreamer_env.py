from functools import partial

import torch
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs.transforms import (
    TransformedEnv,
    Compose,
    ObservationNorm,
    Resize,
    ToTensorImage,
    RewardSum,
    StepCounter,
    FlattenObservation,
    TensorDictPrimer,
    DoubleToFloat
)
from torchrl.envs import ParallelEnv, SerialEnv

from causal_meta.envs.meta_transform import MetaIdxTransform

LIBS = {
    "gym": GymEnv,
    "dm_control": DMControlEnv,
}


def make_dreamer_env(
        env_name,
        env_kwargs=None,
        idx=None,
        task_num=None,
        pixel=True,
        env_library="gym",
        image_size=64,
        variable_num=10,
        state_dim_per_variable=3,
        hidden_dim_per_variable=20,
):
    if env_kwargs is None:
        env_kwargs = {}
    if pixel:
        env_kwargs["from_pixels"] = True
        env_kwargs["pixels_only"] = False

    env = LIBS[env_library](env_name, **env_kwargs)

    transforms = [
        ToTensorImage(),
        Resize(image_size, image_size),
        FlattenObservation(0, -3, allow_positive_dim=True),
        ObservationNorm(0.5, 1.0, standard_normal=True),
        DoubleToFloat(),
        RewardSum(),
        StepCounter(),
    ]

    assert state_dim_per_variable > 0
    default_dict = {
        "state": UnboundedContinuousTensorSpec(
            shape=torch.Size((*env.batch_size, variable_num * state_dim_per_variable))
        ),
        "belief": UnboundedContinuousTensorSpec(
            shape=torch.Size((*env.batch_size, variable_num * hidden_dim_per_variable))
        )
    }
    transforms.append(TensorDictPrimer(random=False, default_value=0, **default_dict))

    if idx is not None:
        transforms.append(MetaIdxTransform(idx, task_num))
    return TransformedEnv(env, transform=Compose(*transforms))


def test_make_pomdp_env():
    torch.multiprocessing.set_sharing_strategy('file_system')

    make_env_fn = partial(make_dreamer_env, env_name="MyCartPole-v0")
    env = make_env_fn()
    td = env.rollout(5, auto_reset=True)

    parallel_env = ParallelEnv(
        num_workers=16,
        create_env_fn=[make_env_fn] * 16,
    )

    td = parallel_env.rollout(5, auto_reset=True)

    parallel_env.close()


if __name__ == '__main__':
    test_make_pomdp_env()
