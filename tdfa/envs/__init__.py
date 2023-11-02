from gym.envs.registration import load_env_plugins as _load_env_plugins
from gym.envs.registration import make, register, registry, spec

# Hook to load plugins from entry points
_load_env_plugins()

# Classic
# ----------------------------------------

register(
    id="CartPoleContinuous-v0",
    entry_point="tdfa.envs.cartpole_continuous:CartPoleContinuousEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)
