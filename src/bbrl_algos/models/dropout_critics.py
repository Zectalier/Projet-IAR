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

class DiscreteDropoutQAgent(NamedCritic):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        name="critic",
        dropout=0.01,
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)
        print("in dropout critic init: state_dim:", state_dim, "hidden_dim:", hidden_dim, "action_dim:", action_dim)
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.Dropout(p=dropout),
            nn.LayerNorm(hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Dropout(p=dropout),
            nn.LayerNorm(hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], action_dim),
        )

        # Show the model structure
        print("model structure:")
        print(self.model)


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