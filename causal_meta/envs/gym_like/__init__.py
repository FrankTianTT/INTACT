from gym.envs.registration import load_env_plugins as _load_env_plugins
from gym.envs.registration import make, register, registry, spec

_load_env_plugins()

register(
    id="MyCartPole-v0",
    entry_point="causal_meta.envs.gym_like.cartpole:CartPoleEnv",
    max_episode_steps=200,
)
register(
    id="MyHalfCheetah-v0",
    entry_point="causal_meta.envs.gym_like.half_cheetah:HalfCheetahEnv",
    max_episode_steps=1000,
)
register(
    id="MyHopper-v0",
    entry_point="causal_meta.envs.gym_like.hopper:HopperEnv",
    max_episode_steps=1000,
)
