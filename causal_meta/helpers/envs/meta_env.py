from functools import partial

import torch
from tensordict import TensorDict
from omegaconf import DictConfig

from causal_meta.helpers.envs.mdp_env import make_mdp_env


def build_make_env_list(env_name, make_env_fn, oracle_context=None):
    if oracle_context is None:
        return [partial(make_env_fn, env_name=env_name)]
    else:
        make_env_list = []
        for idx in range(oracle_context.shape[0]):
            env_kwargs = dict([(key, value.item()) for key, value in oracle_context[idx].items()])
            make_env_list.append(partial(make_env_fn, env_name=env_name, env_kwargs=env_kwargs, idx=idx))
        return make_env_list


def create_make_env_list(cfg: DictConfig, make_env_fn, mode="meta_train"):
    if not cfg.meta:
        return [partial(make_env_fn, env_name=cfg.env_name)], None
    else:
        task_num = cfg.task_num if mode == "meta_train" else cfg.meta_test_task_num
        assert task_num >= 1

        oracle_context = dict(cfg.oracle_context)
        if mode == "meta_test" and cfg.get("new_oracle_context", None):
            oracle_context.update(dict(cfg.new_oracle_context))
        context_dict = {}
        for key, (low, high) in oracle_context.items():
            context_dict[key] = torch.rand(task_num) * (high - low) + low
        oracle_context = TensorDict(context_dict, batch_size=task_num)

        make_env_list = []
        for idx in range(task_num):
            env_kwargs = dict([(key, value[idx].item()) for key, value in context_dict.items()])
            make_env_list.append(
                partial(make_env_fn, env_name=cfg.env_name, env_kwargs=env_kwargs, idx=idx)
            )
        return make_env_list, oracle_context


def test_env_constructor():
    cfg = DictConfig({
        "meta": True,
        "env_name": "MyCartPole-v0",
        "oracle_context": {
            "gravity": (5.0, 20.0),
        },
        "task_num": 50
    })
    make_env_list, oracle_context = create_make_env_list(cfg, make_mdp_env, mode="meta_train")
    env = make_env_list[0]()

    assert oracle_context.shape == (50,)


if __name__ == '__main__':
    test_env_constructor()
