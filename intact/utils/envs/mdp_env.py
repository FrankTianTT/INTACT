from torchrl.envs.libs import GymEnv
from torchrl.envs.transforms import TransformedEnv, Compose, RewardSum, DoubleToFloat, StepCounter
from intact.envs.meta_transform import MetaIdxTransform


def make_mdp_env(
    env_name,
    env_kwargs=None,
    idx=None,
    task_num=None,
    pixel=False,
    env_library="gym",
    max_steps=1000,
):
    """
    Function to create a Markov Decision Process (MDP) environment.

    Args:
        env_name (str): The name of the environment to be created.
        env_kwargs (dict, optional): Additional keyword arguments for the environment. Defaults to None.
        idx (int, optional): Index for the MetaIdxTransform. Defaults to None.
        task_num (int, optional): Task number for the MetaIdxTransform. Defaults to None.
        pixel (bool, optional): If True, the environment will be created from pixels. Defaults to False.
        env_library (str, optional): The library to use for creating the environment. Defaults to "gym".
        max_steps (int, optional): The maximum number of steps for the environment. Defaults to 1000.

    Raises:
        NotImplementedError: If the environment library is "dm_control", as this is not currently implemented.
        ValueError: If the environment library is not recognized.

    Returns:
        TransformedEnv: The created MDP environment, with applied transformations.
    """
    if env_kwargs is None:
        env_kwargs = {}
    if pixel:
        env_kwargs["from_pixels"] = True
        env_kwargs["pixels_only"] = False
    if env_library == "dm_control":
        # env = DMControlEnv(env_name, **env_kwargs)
        raise NotImplementedError
    elif env_library == "gym":
        env = GymEnv(env_name, **env_kwargs, max_episode_steps=max_steps)
        max_steps = None
    else:
        raise ValueError(f"Unknown env library: {env_library}")

    transforms = [DoubleToFloat(), RewardSum(), StepCounter(max_steps)]

    if idx is not None:
        transforms.append(MetaIdxTransform(idx, task_num))
    return TransformedEnv(env, transform=Compose(*transforms))
