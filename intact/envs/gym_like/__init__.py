from gym.envs.registration import load_env_plugins as _load_env_plugins
from gym.envs.registration import register

_load_env_plugins()

register(
    id="MyCartPole-v0",
    entry_point="intact.envs.gym_like.cartpole:CartPoleEnv",
    max_episode_steps=200,
)
register(
    id="MyCartPoleSwingUp-v0",
    entry_point="intact.envs.gym_like.cartpole_swingup:CartPoleSwingUpEnv",
    max_episode_steps=200,
)
register(
    id="MyHalfCheetah-v0",
    entry_point="intact.envs.gym_like.half_cheetah:HalfCheetahEnv",
    max_episode_steps=1000,
)
register(
    id="MyHopper-v0",
    entry_point="intact.envs.gym_like.hopper:HopperEnv",
    max_episode_steps=1000,
)

register(
    id="MultiNode53-v0",
    entry_point="intact.envs.gym_like.multi_node:MultiNodeEnv",
    max_episode_steps=200,
    kwargs=dict(num_rooms=5, context_dim=3, sparsity=0.5),
)

register(
    id="MultiNode53L-v0",
    entry_point="intact.envs.gym_like.multi_node:MultiNodeEnv",
    max_episode_steps=200,
    kwargs=dict(
        num_rooms=5,
        context_dim=3,
        sparsity=0.5,
        context_influence_type="linear",
    ),
)

register(
    id="MultiNode84L-v0",
    entry_point="intact.envs.gym_like.multi_node:MultiNodeEnv",
    max_episode_steps=200,
    kwargs=dict(
        num_rooms=8,
        context_dim=4,
        sparsity=0.5,
        context_influence_type="linear",
    ),
)
