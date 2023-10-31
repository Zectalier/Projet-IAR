import math
import torch
from bbrl.agents import Agent, SeedableAgent, TimeAgent, SerializableAgent
from torch.distributions import Normal


class EGreedyActionSelector(TimeAgent, SeedableAgent, SerializableAgent):
    def __init__(self, epsilon, epsilon_end=None, epsilon_decay=None, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def decay(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    def forward(self, t, **kwargs):
        q_values = self.get(("critic/q_values", t))
        nb_actions = q_values.size()[1]
        size = q_values.size()[0]
        # TODO: make it deterministic if seeded
        is_random = torch.rand(size).lt(self.epsilon).float()
        random_action = torch.randint(low=0, high=nb_actions, size=(size,))
        max_action = q_values.max(1)[1]
        action = is_random * random_action + (1 - is_random) * max_action
        action = action.long()
        self.set(("action", t), action)


class SoftmaxActionSelector(TimeAgent, SeedableAgent, SerializableAgent):
    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def forward(self, t, **kwargs):
        q_values = self.get(("critic/q_values", t))
        probs = torch.softmax(q_values, dim=-1)
        action = torch.distributions.Categorical(probs).sample()
        self.set(("action", t), action)


class RandomDiscreteActor(TimeAgent, SeedableAgent, SerializableAgent):
    def __init__(self, nb_actions, **kwargs):
        super().__init__(**kwargs)
        self.nb_actions = nb_actions

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t))
        size = obs.size()[0]
        action = torch.randint(low=0, high=self.nb_actions, size=(size,))
        self.set(("action", t), action)


class AddGaussianNoise(TimeAgent, SeedableAgent, SerializableAgent):
    def __init__(self, sigma, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        dist = Normal(act, self.sigma)
        action = dist.sample()
        self.set(("action", t), action)


class AddOUNoise(TimeAgent, SeedableAgent, SerializableAgent):
    """
    Ornstein Uhlenbeck process noise for actions as suggested by DDPG paper
    """

    def __init__(self, std_dev, theta=0.15, dt=1e-2, **kwargs):
        super().__init__(**kwargs)
        self.theta = theta
        self.std_dev = std_dev
        self.dt = dt
        self.x_prev = 0

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        x = (
            self.x_prev
            + self.theta * (act - self.x_prev) * self.dt
            + self.std_dev * math.sqrt(self.dt) * torch.randn(act.shape)
        )
        self.x_prev = x
        self.set(("action", t), x)


class KLAgent(TimeAgent, SeedableAgent, SerializableAgent):
    def __init__(self, model_1, model_2, **kwargs):
        super().__init__(**kwargs)
        self.model_1 = model_1
        self.model_2 = model_2

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t))

        dist_1, _ = self.model_1.get_distribution(obs)
        dist_2, _ = self.model_2.get_distribution(obs)
        kl = torch.distributions.kl.kl_divergence(dist_1, dist_2)
        self.set(("kl", t), kl)
