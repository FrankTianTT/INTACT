from typing import Optional, Tuple

import gym
import numpy as np
import pygame
import networkx as nx
import matplotlib.pyplot as plt

import torch
from gym.core import ObsType

from causal_meta.modules.utils import build_mlp


def generate_adjacency_matrix(num_rooms, sparsity):
    # by torch
    adjacency_matrix = torch.zeros(num_rooms, num_rooms)
    for i in range(num_rooms):
        for j in range(num_rooms):
            if i == j:
                adjacency_matrix[i, j] = 0
            else:
                adjacency_matrix[i, j] = torch.rand(1) < sparsity
    return adjacency_matrix


def generate_influence_function(num_rooms):
    func = build_mlp(
        input_dim=num_rooms + 1,
        output_dim=1,
        hidden_dims=[32, 32],
        # hidden_dims=None,
        extra_dims=[num_rooms],
        activate_name='Tanh'
        # activate_name='ELU'
    )

    for name, p in func.named_parameters():
        if 'weight' in name:
            p.data = torch.randn_like(p.data)

    return func


class HeatingEnv(gym.Env):
    def __init__(
            self,
            num_rooms=2,
            sparsity=0.5,
            target_temperature=(18, 22),
            seed=42
    ):
        self.num_rooms = num_rooms
        self.sparsity = sparsity
        self.seed = seed
        assert target_temperature[0] < target_temperature[1]
        self.target_temperature_range = target_temperature

        torch.manual_seed(self.seed)
        self.adjacency_matrix = generate_adjacency_matrix(self.num_rooms, self.sparsity)
        self.inf_func = generate_influence_function(self.num_rooms)

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(num_rooms,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_rooms,), dtype=np.float32)
        self.temperature = np.random.uniform(0, 40, size=(num_rooms,))

        print("Adjacency Matrix:")
        print(self.printing_adjacency_matrix)

    def get_influence(self, action):
        control = torch.from_numpy(action).float().reshape(self.num_rooms, 1, 1)
        temp = torch.from_numpy(self.temperature).float().reshape(1, self.num_rooms)
        temp = temp / 20 - 1  # normalize
        temp = temp.expand(self.num_rooms, -1, -1)  # shape: num_rooms * 1 * num_rooms
        temp = torch.einsum("oi,obi->obi", self.adjacency_matrix, temp)
        inputs = torch.cat([temp, control], dim=-1)  # shape: num_rooms * 1 * (num_rooms + 1)

        with torch.no_grad():
            influence = self.inf_func(inputs)
            influence = torch.clamp(influence, -50, 50) / 20
        influence = influence.squeeze().numpy()

        return influence

    def get_obs(self):
        obs_temp = self.temperature.copy()  # + np.random.normal(0, 0.05, size=(self.num_rooms,))
        return obs_temp / 20 - 1  # normalize

    def step(self, action):
        influence = self.get_influence(action)
        self.temperature += influence
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
    def printing_adjacency_matrix(self):
        string = ""
        for i, out_dim in enumerate(range(self.adjacency_matrix.shape[0])):
            string += " ".join([str(int(ele.item())) for ele in self.adjacency_matrix[out_dim]])
            if i != self.adjacency_matrix.shape[0] - 1:
                string += "\n"
        return string


if __name__ == '__main__':
    env = HeatingEnv()

    # obs, _ = env.reset()
    # for i in range(100):
    #     action = np.random.uniform(-1, 1, size=(env.num_rooms,))
    #     obs, reward, _, _, _ = env.step(action)
    #     print(obs)

    t1, t2 = [], []
    for a1 in np.linspace(-1, 1, 100):
        action = np.array([a1, 0])
        temp = np.random.uniform(0, 40, size=(env.num_rooms,))

        env.temperature = temp.copy()
        new_temp, reward, _, _, _ = env.step(action)
        temp_diff = new_temp - temp
        t1.append(temp_diff[0])
    plt.plot(np.linspace(-1, 1, 100), t1)
    plt.show()

    t2 = []
    for a2 in np.linspace(-1, 1, 100):
        action = np.array([0, a2])
        temp = np.random.uniform(0, 40, size=(env.num_rooms,))

        env.temperature = temp.copy()
        new_temp, reward, _, _, _ = env.step(action)
        temp_diff = new_temp - temp
        t2.append(temp_diff[1])
    plt.plot(np.linspace(-1, 1, 100), t2)
    plt.show()
