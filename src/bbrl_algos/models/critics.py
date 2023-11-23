import numpy as np
import torch
import torch.nn as nn

from bbrl_algos.models.shared_models import build_mlp, build_alt_mlp
from bbrl.agents import TimeAgent, SeedableAgent, SerializableAgent


class NamedCritic(TimeAgent, SeedableAgent, SerializableAgent):
    def __init__(
        self,
        name="critic",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.name = name

    def set_name(self, name: str):
        self.name = name
        return self


class ContinuousQAgent(NamedCritic):
    def __init__(
        self,
        state_dim,
        hidden_layers,
        action_dim,
        name="critic",
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )
        self.is_q_function = True

    def forward(self, t, detach_actions=False):
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        if detach_actions:
            action = action.detach()
        osb_act = torch.cat((obs, action), dim=1)
        q_value = self.model(osb_act)
        self.set((f"{self.name}/q_values", t), q_value)

    def predict_value(self, obs, action):
        obs_act = torch.cat((obs, action), dim=0)
        q_value = self.model(obs_act)
        return q_value


class VAgent(NamedCritic):
    def __init__(
        self,
        state_dim,
        hidden_layers,
        name="critic",
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)
        self.model = build_mlp(
            [state_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )
        self.is_q_function = False

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.model(observation).squeeze(-1)
        self.set((f"{self.name}/v_values", t), critic)


class DiscreteQAgent(NamedCritic):
    def __init__(
        self,
        state_dim,
        hidden_layers,
        action_dim,
        name="critic",
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)
        self.model = build_alt_mlp(
            [state_dim] + list(hidden_layers) + [action_dim], activation=nn.ReLU()
        )
        self.is_q_function = True

    def forward(self, t, choose_action=True, **kwargs):
        obs = self.get(("env/env_obs", t))
        # print("in critic forward: obs:", obs)
        q_values = self.model(obs)
        self.set((f"{self.name}/q_values", t), q_values)
        # Sets the action
        if choose_action:
            action = q_values.argmax(1)
            self.set(("action", t), action)

    def predict_action(self, obs, stochastic):
        q_values = self.model(obs).squeeze(-1)
        if stochastic:
            probs = torch.softmax(q_values, dim=-1)
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = q_values.argmax(-1)
        return action

    def predict_value(self, obs, action):
        q_values = self.model(obs)
        return q_values[action[0].int()]


class TabularQAgent(NamedCritic):
    def __init__(
        self,
        nb_states,
        nb_actions,
        name="critic",
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)
        self.q_table = torch.zeros((nb_states, nb_actions))
        self.is_q_function = True

    def forward(self, t, choose_action=False, **kwargs):
        obs = self.get(("env/env_obs", t))
        q_values = self.q_table[obs, :]
        self.set((f"{self.name}/q_values", t), q_values)
        # Sets the action
        if choose_action:
            action = q_values.argmax()
            self.set(("action", t), action)


class TruncatedQuantileNetwork(NamedCritic):
    def __init__(
        self,
        state_dim,
        hidden_layers,
        n_nets,
        action_dim,
        n_quantiles,
        name="critic",
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)
        self.is_q_function = True
        self.nets = []
        for i in range(n_nets):
            net = build_mlp(
                [state_dim + action_dim] + list(hidden_layers) + [n_quantiles],
                activation=nn.ReLU(),
            )
            self.add_module(f"qf{i}", net)
            self.nets.append(net)
        self.is_q_function = True

    def forward(self, t):
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        obs_act = torch.cat((obs, action), dim=1)
        quantiles = torch.stack(tuple(net(obs_act) for net in self.nets), dim=1)
        self.set((f"{self.name}/quantiles", t), quantiles)
        return quantiles

    def predict_value(self, obs, action):
        obs_act = torch.cat((obs, action), dim=0)
        quantiles = torch.stack(tuple(net(obs_act) for net in self.nets), dim=1)
        return quantiles
