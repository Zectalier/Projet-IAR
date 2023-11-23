from functools import partial
from typing import List
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import gymnasium.spaces as spaces
from bbrl.agents import Agent, Agents, TemporalAgent
from bbrl.workspace import Workspace
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
import pystk2_gymnasium

# Use a a flattened version of the observation and action spaces
# In both case, this corresponds to a dictionary with two keys:
# - `continuous` is a vector corresponding to the continuous observations
# - `discrete` is a vector (of integers) corresponding to discrete observations


class InputModule(nn.Module):
    def __init__(self, space: spaces.Dict, emb_size=10):
        super().__init__()
        discrete: spaces.MultiDiscrete = space["discrete"]
        self.embeddings = nn.Sequential(
            *[nn.Embedding(n, emb_size) for n in discrete.nvec]
        )
        self.size = emb_size * len(discrete.nvec) + np.prod(space["continuous"].shape)

    def forward(self, x: torch.Tensor, values: torch.LongTensor):
        assert (
            len(self.embeddings) == values.shape[1]
        ), f"Number of discrete values ({values.shape[1]}  does not match what was expected ({len(self.embeddings)})"
        inputs = [module(values[:, ix]) for ix, module in enumerate(self.embeddings)]
        inputs.append(x)
        return torch.concatenate(inputs, dim=1)


def build_mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)


class DQNActor(Agent):
    def __init__(self, env):
        super().__init__()
        self.input_module = InputModule(env.observation_space)
        self.actions_ix = [0] + list(env.action_space.nvec.cumsum())
        self.mlp = build_mlp(
            (self.input_module.size, 50, 50, self.actions_ix[-1]), nn.ReLU()
        )

    def forward(self, t, **kwargs):
        x_c = self.get(("env/env_obs/continuous", t))
        x_d = self.get(("env/env_obs/discrete", t))
        y = self.input_module(x_c, x_d)
        y = self.mlp(y)

        for i in range(len(self.actions_ix) - 1):
            action = y[:, self.actions_ix[i] : self.actions_ix[i + 1]]
            self.set((f"action/logits/{i}", t), action)


class ContinuousActor(Agent):
    def __init__(self, env):
        super().__init__()
        self.input_module = InputModule(env.observation_space)
        self.mlp = build_mlp(
            (self.input_module.size, 50, 50, env.action_space.shape[0]), nn.ReLU()
        )

    def forward(self, t, **kwargs):
        x_c = self.get(("env/env_obs/continuous", t))
        x_d = self.get(("env/env_obs/discrete", t))
        y = self.input_module(x_c, x_d)
        action = self.mlp(y)

        self.set(("action", t), action)


class RandomActor(Agent):
    def __init__(self, space):
        super().__init__()
        self.space = space

    def forward(self, t: int):
        # self.set(("action", t), torch.Tensor(self.space.sample()).unsqueeze(0))
        pass


class ActionCopier(Agent):
    def forward(self, t: int):
        action = self.get(("env/env_obs/action", t))
        self.set(("action", t), action)
