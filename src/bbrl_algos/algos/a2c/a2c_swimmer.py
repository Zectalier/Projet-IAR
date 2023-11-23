import sys
import os

import optuna
import yaml
import mujoco_py

from omegaconf import DictConfig
from bbrl import get_arguments, get_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from bbrl.utils.functional import gae

import hydra

import torch
import torch.nn as nn

from bbrl_algos.models.stochastic_actors import TunableVarianceContinuousActor
from bbrl_algos.models.stochastic_actors import SquashedGaussianActor
from bbrl_algos.models.stochastic_actors import StateDependentVarianceContinuousActor
from bbrl_algos.models.stochastic_actors import ConstantVarianceContinuousActor
from bbrl_algos.models.stochastic_actors import DiscreteActor, BernoulliActor
from bbrl_algos.models.critics import VAgent
from bbrl_algos.models.loggers import Logger
from bbrl.utils.chrono import Chrono

from bbrl.visu.plot_policies import plot_policy
from bbrl.visu.plot_critics import plot_critic
from bbrl_algos.models.envs import get_env_agents
from bbrl_algos.models.hyper_params import launch_optuna
from bbrl_algos.models.utils import save_best

# HYDRA_FULL_ERROR = 1

import matplotlib

matplotlib.use("TkAgg")


# Create the A2C Agent
def create_a2c_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    action_agent = globals()[cfg.algorithm.actor_type](
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
    )
    tr_agent = Agents(train_env_agent, action_agent)
    ev_agent = Agents(eval_env_agent, action_agent)

    critic_agent = TemporalAgent(
        VAgent(obs_size, cfg.algorithm.architecture.critic_hidden_size)
    )

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    return train_agent, eval_agent, critic_agent


# Configure the optimizer over the a2c agent
def setup_optimizers(cfg, action_agent, critic_agent):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = nn.Sequential(action_agent, critic_agent).parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer


def compute_advantages_loss(cfg, reward, must_bootstrap, v_value):
    # Compute temporal difference
    reward = reward[1]
    next_val = v_value[1]
    mb = must_bootstrap[1]
    current_val = v_value[0]
    advantages = gae(
        reward,
        next_val,
        mb,
        current_val,
        cfg.algorithm.discount_factor,
        cfg.algorithm.gae,
    )
    td = advantages - current_val
    td_error = td**2
    critic_loss = td_error.mean()
    return critic_loss, advantages


def compute_actor_loss(action_logp, td):
    a2c_loss = action_logp[:-1] * td.detach()
    return a2c_loss.mean()


def run_a2c(cfg, logger, trial=None):
    chrono = Chrono()
    best_reward = float("-inf")

    # 2) Create the environment agent
    train_env_agent, eval_env_agent = get_env_agents(cfg)

    # 3) Create the A2C Agent
    a2c_agent, eval_agent, critic_agent = create_a2c_agent(
        cfg, train_env_agent, eval_env_agent
    )

    # 5) Configure the workspace to the right dimension
    # Note that no parameter is needed to create the workspace.
    # In the training loop, calling the agent() and critic_agent()
    # will take the workspace as parameter
    train_workspace = Workspace()  # Used for training

    # 6) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(cfg, a2c_agent, critic_agent)
    nb_steps = 0
    tmp_steps = 0

    # 7) Training loop
    while nb_steps < cfg.algorithm.n_steps:
        # Execute the agent in the workspace
        if nb_steps > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            a2c_agent(
                train_workspace,
                t=1,
                n_steps=cfg.algorithm.n_steps_train - 1,
                stochastic=True,
                predict_proba=False,
                compute_entropy=True,
            )
        else:
            a2c_agent(
                train_workspace,
                t=0,
                n_steps=cfg.algorithm.n_steps_train,
                stochastic=True,
                predict_proba=False,
                compute_entropy=True,
            )

        # Compute the critic value over the whole workspace
        critic_agent(train_workspace, n_steps=cfg.algorithm.n_steps_train)

        transition_workspace = train_workspace.get_transitions()

        v_value, terminated, reward, action, action_logp = transition_workspace[
            "critic/v_values",
            "env/terminated",
            "env/reward",
            "action",
            "policy/action_logprobs",
        ]

        nb_steps += action[0].shape[0]
        # Determines whether values of the critic should be propagated
        # True if the task was not terminated.
        must_bootstrap = ~terminated[1]

        # Compute critic loss
        critic_loss, advantages = compute_advantages_loss(
            cfg, reward, must_bootstrap, v_value
        )
        a2c_loss = compute_actor_loss(action_logp, advantages)

        # Compute entropy loss
        entropy_loss = torch.mean(train_workspace["entropy"])

        # Store the losses for tensorboard display
        logger.log_losses(nb_steps, critic_loss, entropy_loss, a2c_loss)

        # Compute the total loss
        loss = (
            cfg.algorithm.critic_coef * critic_loss
            - cfg.algorithm.entropy_coef * entropy_loss
            - cfg.algorithm.a2c_coef * a2c_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            a2c_agent.parameters(), cfg.algorithm.max_grad_norm
        )
        optimizer.step()

        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace,
                t=0,
                stop_variable="env/done",
                stochastic=False,
                predict_proba=False,
                render=False,  # render=True,
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.log_reward_losses(rewards, nb_steps)
            print(
                f"nb_steps: {nb_steps}, reward: {mean:.0f}, best_reward: {best_reward:.0f}"
            )
            if mean > best_reward:
                best_reward = mean
                if cfg.save_best:
                    policy = eval_agent.agent.agents[1]
                    save_best(
                        policy, cfg.gym_env.env_name, mean, "./a2c_best_agents/", "a2c"
                    )
                    critic = critic_agent.agent
                    if cfg.plot_agents:
                        plot_policy(
                            policy,
                            eval_env_agent,
                            best_reward,
                            "./a2c_plots/",
                            cfg.gym_env.env_name,
                            stochastic=False,
                        )
                        plot_critic(
                            critic,
                            eval_env_agent,
                            best_reward,
                            "./a2c_plots/",
                            cfg.gym_env.env_name,
                        )
            if trial is not None:
                trial.report(mean, nb_steps)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    chrono.stop()
    return best_reward


@hydra.main(
    config_path="./configs/",
    config_name="a2c_swimmer.yaml",
    # version_base="1.1",
)
def main(cfg_raw: DictConfig):
    torch.random.manual_seed(seed=cfg_raw.algorithm.seed.torch)

    if "optuna" in cfg_raw:
        launch_optuna(cfg_raw, run_a2c)
    else:
        logger = Logger(cfg_raw)
        run_a2c(cfg_raw, logger)
