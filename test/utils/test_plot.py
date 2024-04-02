import os
import tempfile

from intact.utils.models.mdp import make_mdp_model, MDPConfig
import numpy as np

from intact.utils.envs.mdp_env import make_mdp_env
from intact.utils.envs.meta_env import create_make_env_list
from intact.utils.plot import plot_context

tmp_dir = tempfile.gettempdir()


def test_plot_context1():
    task_num = 5

    config = MDPConfig()
    config.env_name = "MyCartPole-v0"
    config.oracle_context = {"gravity": (5.0, 20.0)}
    config.meta = True
    config.task_num = task_num

    make_env_list, oracle_context = create_make_env_list(config, make_mdp_env, mode="meta_train")

    env = make_mdp_env(config.env_name)
    world_model, model_based_env = make_mdp_model(config, env)

    log_prefix = os.path.join(tmp_dir, "test_plot_context1")
    plot_context(config, world_model, oracle_context, log_prefix=log_prefix)


def test_plot_context2():
    task_num = 5

    config = MDPConfig()
    config.env_name = "MyCartPole-v0"
    config.oracle_context = {"gravity": (5.0, 20.0), "cart_vel_bias": (-1.0, 1.0)}
    config.meta = True
    config.task_num = task_num

    make_env_list, oracle_context = create_make_env_list(config, make_mdp_env, mode="meta_train")

    env = make_mdp_env(config.env_name)
    world_model, model_based_env = make_mdp_model(config, env)

    log_prefix = os.path.join(tmp_dir, "test_plot_context2")
    plot_context(config, world_model, oracle_context, log_prefix=log_prefix, color_values=np.random.randn(task_num))
