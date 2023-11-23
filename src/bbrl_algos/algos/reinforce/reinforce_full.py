import os
import optuna
import yaml
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from bbrl.agents.gymnasium import make_env, GymAgent, ParallelGymAgent
from functools import partial

from bbrl import get_arguments, get_class, instantiate_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent, PrintAgent
from bbrl.agents.agent import Agent

from bbrl_algos.models.stochastic_actors import TunableVarianceContinuousActor
from bbrl_algos.models.stochastic_actors import StateDependentVarianceContinuousActor
from bbrl_algos.models.stochastic_actors import ConstantVarianceContinuousActor
from bbrl_algos.models.stochastic_actors import DiscreteActor, BernoulliActor
from bbrl_algos.models.critics import VAgent
from bbrl_algos.models.loggers import Logger
from bbrl_algos.models.hyper_params import launch_optuna
from bbrl_algos.models.utils import save_best
from bbrl_algos.models.envs import get_eval_env_agent
from bbrl.utils.chrono import Chrono

from bbrl.visu.plot_policies import plot_policy
from bbrl.visu.plot_critics import plot_critic

# HYDRA_FULL_ERROR = 1
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


def apply_sum(reward):
    # print(reward)
    reward_sum = reward.sum(axis=0)
    # print("sum", reward_sum)
    for i in range(len(reward)):
        reward[i] = reward_sum
    # print("final", reward)
    return reward


def apply_discounted_sum(cfg, reward):
    # print(reward)
    tmp = 0
    for i in reversed(range(len(reward))):
        reward[i] = reward[i] + cfg.algorithm.discount_factor * tmp
        tmp = reward[i]
    # print("final", reward)
    return reward


def apply_discounted_sum_minus_baseline(cfg, reward, baseline):
    # print(reward)
    tmp = 0
    for i in reversed(range(len(reward))):
        reward[i] = reward[i] + cfg.algorithm.discount_factor * tmp - baseline[i]
        tmp = reward[i]
    return reward


# Create the REINFORCE Agent
def create_reinforce_agent(cfg, env_agent):
    obs_size, act_size = env_agent.get_obs_and_actions_sizes()
    action_agent = action_agent = globals()[cfg.algorithm.actor_type](
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
    )
    # print_agent = PrintAgent()
    tr_agent = Agents(env_agent, action_agent)
    critic_agent = TemporalAgent(
        VAgent(obs_size, cfg.algorithm.architecture.critic_hidden_size)
    )

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    return train_agent, critic_agent


# Configure the optimizer over the a2c agent
def setup_optimizer(cfg, actor, critic):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = nn.Sequential(actor, critic).parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer


def compute_critic_loss(cfg, reward, must_bootstrap, v_value):
    # Compute temporal difference
    # print(f"reward:{reward}, V:{v_value}, MB:{must_bootstrap}")
    target = (
        reward[1:]
        + cfg.algorithm.discount_factor
        * v_value[1:].detach()
        * must_bootstrap[1:].int()
    )
    td = target - v_value[:-1]

    # Compute critic loss
    td_error = td**2
    critic_loss = td_error.mean()
    # print(f"target:{target}, td:{td}, cl:{critic_loss}")
    return critic_loss


def compute_actor_loss(action_logprob, reward, must_bootstrap):
    actor_loss = action_logprob * reward.detach() * must_bootstrap.int()
    return actor_loss.mean()


def run_reinforce(cfg, logger, trial=None):
    best_reward = float("-inf")

    # 2) Create the environment agent
    env_agent = get_eval_env_agent(cfg)

    reinforce_agent, critic_agent = create_reinforce_agent(cfg, env_agent)

    # 7) Configure the optimizer over the a2c agent
    optimizer = setup_optimizer(cfg, reinforce_agent, critic_agent)

    # 8) Training loop
    nb_steps = 0

    for episode in range(cfg.algorithm.nb_episodes):
        # print_agent.reset()
        # Execute the agent on the workspace to sample complete episodes
        # Since not all the variables of workspace will be overwritten, it is better to clear the workspace
        # Configure the workspace to the right dimension.
        train_workspace = Workspace()

        reinforce_agent(
            train_workspace,
            stochastic=True,
            t=0,
            stop_variable="env/done",
            compute_entropy=True,
        )
        # Get relevant tensors (size are timestep x n_envs x ....)
        terminated, action_logprobs, reward, action = train_workspace[
            "env/terminated",
            "policy/action_logprobs",
            "env/reward",
            "action",
        ]
        critic_agent(train_workspace, stop_variable="env/done")
        v_value = train_workspace["critic/v_values"]
        for i in range(cfg.algorithm.n_envs_eval):
            nb_steps += len(action[:, i])

        # Determines whether values of the critic should be propagated
        must_bootstrap = ~terminated

        critic_loss = compute_critic_loss(cfg, reward, must_bootstrap, v_value)

        # reward = apply_sum(reward)
        reward = apply_discounted_sum(cfg, reward)
        # reward = apply_discounted_sum_minus_baseline(cfg, reward, v_value)
        actor_loss = compute_actor_loss(action_logprobs, reward, must_bootstrap)

        entropy_loss = torch.mean(train_workspace["entropy"])
        # Log losses
        logger.log_losses(nb_steps, critic_loss, entropy_loss, actor_loss)

        loss = (
            -cfg.algorithm.entropy_coef * entropy_loss
            + cfg.algorithm.critic_coef * critic_loss
            - cfg.algorithm.actor_coef * actor_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the cumulated reward on final_state
        rewards = train_workspace["env/cumulated_reward"][-1]
        mean = rewards.mean()
        logger.log_reward_losses(rewards, nb_steps)
        print(f"episode: {episode}, reward: {mean}")

        if cfg.save_best and mean > best_reward:
            best_reward = mean
            policy = reinforce_agent.agent.agents[1]
            critic = critic_agent.agent
            save_best(
                policy,
                cfg.gym_env.env_name,
                mean,
                "./reinforce_best_agents/",
                "reinforce",
            )
            if cfg.plot_agents:
                plot_policy(
                    policy,
                    env_agent,
                    best_reward,
                    "./reinforce_plots/",
                    cfg.gym_env.env_name,
                    stochastic=False,
                )
                plot_critic(
                    critic,
                    env_agent,
                    best_reward,
                    "./reinforce_plots/",
                    cfg.gym_env.env_name,
                )


@hydra.main(
    config_path="./configs/",
    # config_name="reinforce_debugv.yaml",
    config_name="reinforce_cartpole.yaml",
    # version_base="1.1",
)
def main(cfg_raw: DictConfig):
    torch.random.manual_seed(seed=cfg_raw.algorithm.seed.torch)

    if "optuna" in cfg_raw:
        launch_optuna(cfg_raw, run_reinforce)
    else:
        logger = Logger(cfg_raw)
        run_reinforce(cfg_raw, logger)


if __name__ == "__main__":
    main()
