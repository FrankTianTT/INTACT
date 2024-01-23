from typing import Optional, Tuple
from collections import defaultdict

import gym
import numpy as np
import pygame
import networkx as nx
import matplotlib.pyplot as plt

import torch
from gym.core import ObsType

from causal_meta.modules.utils import build_mlp
from causal_meta.utils.graph import check_structural_sparsity


def generate_graph_cross_rooms(num_rooms, sparsity):
    adjacency_matrix = torch.zeros(num_rooms, num_rooms).int()
    for i in range(num_rooms):
        for j in range(num_rooms):
            if i == j:
                adjacency_matrix[i, j] = 0
            else:
                adjacency_matrix[i, j] = torch.rand(1) < sparsity
    return adjacency_matrix


def generate_graph_between_room_and_context(num_rooms, context_dim, sparsity):
    graph = torch.lt(torch.rand(num_rooms, context_dim), sparsity).int()

    while not check_structural_sparsity(graph.numpy()):
        graph = torch.lt(torch.rand(num_rooms, context_dim), sparsity).int()

    return graph


def generate_influence_function(num_rooms, context_dim):
    func = build_mlp(num_rooms + 1 + context_dim, 1, [32, 32], num_rooms, activate_name='Tanh')

    for name, p in func.named_parameters():
        if 'weight' in name:
            p.data = torch.randn_like(p.data)

    return func


class HeatingEnv(gym.Env):
    def __init__(
            self,
            num_rooms=5,
            context_dim=3,
            sparsity=0.5,
            context_sparsity=0.3,
            target_temperature=(18, 22),
            dt=0.1,
            seed=42,
            context_influence_type="neural",
            frameskip=1,
            render_mode='rgb_array',
            **context_kwargs,
    ):
        assert context_dim <= num_rooms, "source variables should be less than observed variables"
        self.num_rooms = num_rooms
        self.context_dim = context_dim
        self.sparsity = sparsity
        self.context_sparsity = context_sparsity
        assert target_temperature[0] < target_temperature[1]
        self.target_temperature_range = target_temperature
        self.dt = dt
        self.seed = seed
        self.context_influence_type = context_influence_type
        self.contexts = torch.zeros(context_dim)
        for i in range(self.context_dim):
            if f"c{i + 1}" in context_kwargs:
                self.contexts[i] = context_kwargs[f"c{i + 1}"]
                del context_kwargs[f"c{i + 1}"]
        # assert len(context_kwargs) == 0, f"Unknown context variables: {context_kwargs}"

        torch.manual_seed(self.seed)
        self.room_graph = generate_graph_cross_rooms(self.num_rooms, self.sparsity)
        self.context_graph = generate_graph_between_room_and_context(
            self.num_rooms, self.context_dim, self.context_sparsity
        )

        self.masked_context = self.context_graph * self.contexts.expand(self.num_rooms, -1)

        self.inf_func = generate_influence_function(
            num_rooms=self.num_rooms,
            context_dim=self.context_dim if self.context_influence_type == "neural" else 0
        )
        self.influence = None

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(num_rooms,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_rooms,), dtype=np.float32)
        self.temperature = np.random.uniform(0, 40, size=(num_rooms,))

    @property
    def total_graph(self):
        action_graph = torch.eye(self.num_rooms).int()
        return torch.cat([self.room_graph, action_graph, self.context_graph], dim=-1)

    def calculate_influence(self, action):
        control = torch.from_numpy(action).float().reshape(self.num_rooms, 1)
        temp = torch.from_numpy(self.temperature).float().reshape(1, self.num_rooms)
        temp = (temp - 20) / 20
        temp = self.room_graph * temp.expand(self.num_rooms, -1)
        if self.context_influence_type == "neural":
            inputs = torch.cat([temp, control, self.masked_context], dim=-1)
        else:
            inputs = torch.cat([temp, control], dim=-1)

        with torch.no_grad():
            influence = self.inf_func(inputs.reshape(self.num_rooms, 1, -1))
        self.influence = influence.squeeze().numpy()

        if self.context_influence_type == "linear":
            self.influence += self.masked_context.mean(dim=-1).numpy() * 5.
        elif self.context_influence_type == "tanh":
            self.influence += torch.tanh(self.masked_context.mean(dim=-1)).numpy()
        elif self.context_influence_type == "neural":
            pass
        else:
            raise NotImplementedError

    def get_obs(self):
        obs_temp = self.temperature.copy()  # + np.random.normal(0, 0.05, size=(self.num_rooms,))
        return (obs_temp - 20) / 20  # normalize

    def step(self, action):
        self.calculate_influence(action)
        self.temperature += self.influence * self.dt
        # self.temperature = np.clip(self.temperature, 0, 40)

        reward = - np.abs(self.temperature - 20).mean()

        return self.get_obs(), reward, False, False, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        self.temperature = np.random.uniform(0, 40, size=(self.num_rooms,))
        return self.get_obs(), {}

    @property
    def printing_total_graph(self):
        string = ""
        for i, out_dim in enumerate(range(self.total_graph.shape[0])):
            string += " ".join([str(int(ele.item())) for ele in self.total_graph[out_dim]])
            if i != self.total_graph.shape[0] - 1:
                string += "\n"
        return string


if __name__ == '__main__':
    # 0 0 0 1 0 1 0 0 0 0 1 1 0
    # 1 0 0 1 0 0 1 0 0 0 1 0 0
    # 0 1 0 0 0 0 0 1 0 0 0 1 1
    # 0 0 0 0 1 0 0 0 1 0 0 0 0
    # 0 0 1 0 0 0 0 0 0 1 0 0 1

    inf = []
    for c1 in np.linspace(-1, 1, 100):
        env = HeatingEnv(c1=c1)

        action = np.zeros(env.num_rooms)
        # action[0] = a1
        temp = np.zeros(env.num_rooms)
        # temp[3] = a1 * 20 + 20

        env.temperature = temp.copy()
        env.step(action)

        inf.append(env.influence[0])
    plt.plot(np.linspace(-1, 1, 100), inf)
    plt.show()
    #
    # t2 = []
    # for a2 in np.linspace(-1, 1, 100):
    #     action = np.array([0, a2])
    #     temp = np.random.uniform(0, 40, size=(env.num_rooms,))
    #
    #     env.temperature = temp.copy()
    #     new_temp, reward, _, _, _ = env.step(action)
    #     temp_diff = new_temp - temp
    #     t2.append(temp_diff[1])
    # plt.plot(np.linspace(-1, 1, 100), t2)
    # plt.show()

    # env = HeatingEnv(c1=2, c2=-1)
    # print(env.printing_total_graph)
    # 0 0 0 1 0 1 0 0 0 0 1 1 0
    # 1 0 0 1 0 0 1 0 0 0 1 0 0
    # 0 1 0 0 0 0 0 1 0 0 0 1 1
    # 0 0 0 0 1 0 0 0 1 0 0 0 0
    # 0 0 1 0 0 0 0 0 0 1 0 0 1
