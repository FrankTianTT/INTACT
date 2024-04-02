from abc import abstractmethod

import torch.nn as nn

from intact.modules.models.context_model import ContextModel


class BaseWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        meta=False,
        max_context_dim=10,
        task_num=100,
        residual=True,
        learned_reward=True,
        learned_termination=True,
    ):
        """World-model class for environment learning with causal discovery.

        :param obs_dim: number of observation dimensions
        :param action_dim: number of action dimensions
        :param meta: whether to use envs-RL
        :param max_context_dim: number of context dimensions, used for envs-RL, set to 0 if normal RL
        :param task_num: number of tasks, used for envs-RL, set to 0 if normal RL
        :param residual: whether to use residual connection for transition model
        :param learned_reward: whether to learn reward model
        :param learned_termination: whether to learn termination model
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.meta = meta
        self.max_context_dim = max_context_dim if meta else 0
        self.task_num = task_num
        self.residual = residual
        self.learned_reward = learned_reward
        self.learned_termination = learned_termination

        # context model
        self.context_model = ContextModel(meta=meta, max_context_dim=max_context_dim, task_num=task_num)

        # mdp model (transition + reward + termination)
        self.nets = self.build_nets()

    @property
    @abstractmethod
    def learn_obs_var(self):
        raise NotImplementedError

    @abstractmethod
    def build_nets(self) -> nn.ModuleDict:
        raise NotImplementedError

    @property
    def params_dict(self):
        return dict(nets=self.nets.parameters(), context=self.context_model.parameters())

    def get_parameter(self, target: str):
        if target in self.params_dict:
            return self.params_dict[target]
        else:
            raise NotImplementedError

    @property
    def all_input_dim(self):
        return self.obs_dim + self.action_dim + self.max_context_dim

    @property
    def output_dim(self):
        return self.obs_dim + int(self.learned_reward) + int(self.learned_termination)

    def reset(self, task_num=None):
        self.context_model.reset(task_num)

    def freeze_module(self):
        for name in self.params_dict.keys():
            if name == "context":
                continue
            for param in self.params_dict[name]:
                param.requires_grad = False
