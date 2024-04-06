from functools import partial

from torchrl.envs import GymEnv, TransformedEnv, SerialEnv, ParallelEnv
import torch
from tqdm import tqdm

from intact.envs.meta_transform import MetaIdxTransform


def gen_mdp_data(env_name="CartPoleContinuous-v0", sample_num=1000):
    env = GymEnv(env_name)

    td = env.rollout(sample_num, auto_reset=True, break_when_any_done=False)

    obs = td["observation"]
    action = td["action"]
    next_obs = td[("next", "observation")]

    return obs, action, next_obs


def env_constructor(
    env_name="CartPoleContinuous-v0",
    task_num=100,
    oracle_context=None,
):
    if oracle_context is None:
        if env_name == "CartPoleContinuous-v0":
            # oracle_context = dict(gravity=(1, 10))
            oracle_context = dict(x_dot_bias=(-2, 2))
        else:
            raise NotImplementedError(
                "oracle_context is None, but env_name is {}".format(env_name)
            )

    def make_env(gym_kwargs, idx):
        if gym_kwargs is None:
            gym_kwargs = {}
        env = GymEnv(env_name, **gym_kwargs)
        return TransformedEnv(env, transform=MetaIdxTransform(idx, task_num))

    context_dict = {}
    for key, (low, high) in oracle_context.items():
        context_dict[key] = torch.rand(task_num) * (high - low) + low

    make_env_list = []
    for idx in range(task_num):
        gym_kwargs = dict(
            [(key, value[idx].item()) for key, value in context_dict.items()]
        )
        make_env_list.append(partial(make_env, gym_kwargs=gym_kwargs, idx=idx))
    return make_env_list, context_dict


def gen_meta_mdp_data(
    env_name="CartPoleContinuous-v0", task_num=100, sample_num=10000
):
    if env_name == "toy":
        context_dict = {"p0": torch.rand(task_num), "p1": torch.rand(task_num)}

        obs = torch.randn(sample_num, 2)
        action = torch.randn(sample_num, 1)
        idx = torch.randint(0, task_num, (sample_num, 1))
        next_obs = obs.clone()
        next_obs[:, 0] += +torch.sin(
            context_dict["p0"][idx.squeeze()] * 2 * torch.pi
        )
        next_obs[:, 1] += +torch.cos(
            context_dict["p1"][idx.squeeze()] * 2 * torch.pi
        )

        return obs, action, next_obs, idx, context_dict

    make_env_list, context_dict = env_constructor(
        env_name=env_name, task_num=task_num
    )
    env = SerialEnv(len(make_env_list), make_env_list, shared_memory=False)

    par = tqdm(range(sample_num), desc="Generating data")

    def callback(env, tensor_dict):
        par.update(tensor_dict.numel())

    td = env.rollout(
        sample_num // task_num,
        auto_reset=True,
        break_when_any_done=False,
        callback=callback,
    )

    td = td.reshape(-1)
    idx = td["idx"]
    obs = td["observation"]
    action = td["action"]
    next_obs = td[("next", "observation")]

    return obs, action, next_obs, idx, context_dict


if __name__ == "__main__":
    gen_meta_mdp_data("toy")
