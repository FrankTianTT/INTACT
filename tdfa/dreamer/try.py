import gym
import hydra
import torch
from torchrl.envs import GymWrapper, TransformedEnv, RewardSum, DoubleToFloat, Compose
from torchrl.trainers.helpers.envs import transformed_env_constructor
from torchrl.trainers.helpers.envs import (
    correct_for_frame_skip,
    initialize_observation_norm_transforms,
    retrieve_observation_norms_state_dict,
)

import tdfa
from tdfa.dreamer.helper import make_causal_dreamer


@hydra.main(version_base="1.1", config_path=".", config_name="config")
def main(cfg):
    if not torch.cuda.is_available():
        cfg.model_device = "cpu"
        cfg.collector_device = "cpu"
    print("Using device {}".format(cfg.model_device))

    make_env = transformed_env_constructor(cfg)
    proof_env = make_env()

    initialize_observation_norm_transforms(
        proof_environment=proof_env, num_iter=cfg.init_env_steps, key=("next", "pixels")
    )

    _, obs_norm_state_dict = retrieve_observation_norms_state_dict(proof_env)[0]

def make_env(env_name):
    gym_env = gym.make(env_name, render_mode="rgb_array")
    # get BaseMujocoEnv

    env = GymWrapper(gym_env, from_pixels=True, pixels_only=True)
    transform = Compose(
        DoubleToFloat(),
        # MetaIdxTransform(idx, cfg.task_num),
        RewardSum()
    )
    return TransformedEnv(env, transform=transform)


if __name__ == '__main__':
    main()
