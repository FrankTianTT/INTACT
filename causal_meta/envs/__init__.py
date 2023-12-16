from gym.envs.registration import load_env_plugins as _load_env_plugins
from gym.envs.registration import make, register, registry, spec

from causal_meta.envs.termination_fns import termination_fns_dict
from causal_meta.envs.reward_fns import reward_fns_dict

# Hook to load plugins from entry points
_load_env_plugins()

# Classic
# ----------------------------------------

register(
    id="CartPoleContinuous-v0",
    entry_point="causal_meta.envs.cartpole_continuous:CartPoleContinuousEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)
