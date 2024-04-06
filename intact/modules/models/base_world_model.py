from abc import abstractmethod

import torch.nn as nn

from intact.modules.models.context_model import ContextModel


class BaseWorldModel(nn.Module):
    """
    Abstract base class for a world model in an environment learning system with causal discovery.

    This class should be subclassed and the abstract methods implemented.
    """

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
        """
        Initialize the BaseWorldModel.

        Args:
            obs_dim (int): The number of observation dimensions.
            action_dim (int): The number of action dimensions.
            meta (bool, optional): Whether to use envs-RL. Defaults to False.
            max_context_dim (int, optional): The number of context dimensions, used for envs-RL, set to 0 if normal RL. Defaults to 10.
            task_num (int, optional): The number of tasks, used for envs-RL, set to 0 if normal RL. Defaults to 100.
            residual (bool, optional): Whether to use residual connection for transition model. Defaults to True.
            learned_reward (bool, optional): Whether to learn reward model. Defaults to True.
            learned_termination (bool, optional): Whether to learn termination model. Defaults to True.
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
        self.context_model = ContextModel(
            meta=meta, max_context_dim=max_context_dim, task_num=task_num
        )

        # mdp model (transition + reward + termination)
        self.nets = self.build_nets()

    @property
    @abstractmethod
    def learn_obs_var(self):
        """
        Abstract property for learning observation variance.

        This property should be implemented in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def build_nets(self) -> nn.ModuleDict:
        """
        Abstract method for building the networks.

        This method should be implemented in subclasses.
        """
        raise NotImplementedError

    @property
    def params_dict(self):
        """
        Get a dictionary of the parameters.

        Returns:
            dict: A dictionary of the parameters.
        """
        return dict(
            nets=self.nets.parameters(),
            context=self.context_model.parameters(),
        )

    def get_parameter(self, target: str):
        """
        Get the parameters of the specified target.

        Args:
            target (str): The target to get the parameters of.

        Returns:
            iterator: An iterator over the parameters of the target.

        Raises:
            NotImplementedError: If the target is not recognized.
        """
        if target in self.params_dict:
            return self.params_dict[target]
        else:
            raise NotImplementedError

    @property
    def all_input_dim(self):
        """
        Get the total input dimension.

        Returns:
            int: The total input dimension.
        """
        return self.obs_dim + self.action_dim + self.max_context_dim

    @property
    def output_dim(self):
        """
        Get the output dimension.

        Returns:
            int: The output dimension.
        """
        return (
            self.obs_dim
            + int(self.learned_reward)
            + int(self.learned_termination)
        )

    def reset(self, task_num=None):
        """
        Reset the context model.

        Args:
            task_num (int, optional): The number of tasks. Defaults to None.
        """
        self.context_model.reset(task_num)

    def freeze_module(self):
        """
        Freeze the parameters of the module.
        """
        for name in self.params_dict.keys():
            if name == "context":
                continue
            for param in self.params_dict[name]:
                param.requires_grad = False
