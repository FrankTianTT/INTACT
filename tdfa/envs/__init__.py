from gym.envs.registration import load_env_plugins as _load_env_plugins
from gym.envs.registration import make, register, registry, spec

from tdfa.envs import termination_fns, reward_fns

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

termination_fns_dict = {
    "cartpole": termination_fns.cartpole,
    "inverted_pendulum": termination_fns.inverted_pendulum,
    "no_termination": termination_fns.no_termination,
    # "walker2d": termination_fns.walker2d,
    # "ant": termination_fns.ant,
}
reward_fns_dict = {
    "ones": reward_fns.ones,
}
