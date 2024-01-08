import numpy as np
import torch
import torch.nn as nn

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
        dropout=0.0001,
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_layers[0]),
            nn.Dropout(p=dropout),
            nn.LayerNorm(hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.Dropout(p=dropout),
            nn.LayerNorm(hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], 1)
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