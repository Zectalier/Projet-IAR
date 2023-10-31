import sys
import os
import copy
import numpy as np

import torch
import torch.nn as nn
import hydra
import optuna

from omegaconf import DictConfig
from bbrl import get_arguments, get_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl.utils.chrono import Chrono

from bbrl_algos.models.loggers import Logger

from bbrl.visu.plot_policies import plot_policy
from bbrl.visu.plot_critics import plot_critic

from bbrl_algos.models.actors import ContinuousDeterministicActor
from bbrl_algos.models.critics import ContinuousQAgent
from bbrl_algos.models.plotters import Plotter
from bbrl_algos.models.exploration_agents import AddGaussianNoise
from bbrl_algos.models.envs import get_env_agents
from bbrl_algos.models.hyper_params import launch_optuna
from bbrl_algos.models.utils import save_best

# HYDRA_FULL_ERROR = 1
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# Create the DDPG Agent
def create_ddpg_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    critic = ContinuousQAgent(
        obs_size,
        cfg.algorithm.architecture.critic_hidden_size,
        act_size,
        seed=cfg.algorithm.seed.q,
    )
    target_critic = copy.deepcopy(critic).set_name("target-critic")
    actor = ContinuousDeterministicActor(
        obs_size,
        cfg.algorithm.architecture.actor_hidden_size,
        act_size,
        seed=cfg.algorithm.seed.act,
    )
    # target_actor = copy.deepcopy(actor)
    noise_agent = AddGaussianNoise(
        cfg.algorithm.action_noise,
        seed=cfg.algorithm.seed.explorer,
    )
    tr_agent = Agents(train_env_agent, actor, noise_agent)  # TODO : add OU noise
    ev_agent = Agents(eval_env_agent, actor)

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    return train_agent, eval_agent, actor, critic, target_critic  # , target_actor


# Configure the optimizer
def setup_optimizers(cfg, actor, critic):
    actor_optimizer_args = get_arguments(cfg.actor_optimizer)
    parameters = actor.parameters()
    actor_optimizer = get_class(cfg.actor_optimizer)(parameters, **actor_optimizer_args)
    critic_optimizer_args = get_arguments(cfg.critic_optimizer)
    parameters = critic.parameters()
    critic_optimizer = get_class(cfg.critic_optimizer)(
        parameters, **critic_optimizer_args
    )
    return actor_optimizer, critic_optimizer


def compute_critic_loss(cfg, reward, must_bootstrap, q_values, target_q_values):
    # Compute temporal difference
    q_next = target_q_values
    target = (
        reward[:-1].squeeze()
        + cfg.algorithm.discount_factor * q_next.squeeze(-1) * must_bootstrap.int()
    )
    mse = nn.MSELoss()
    critic_loss = mse(target, q_values.squeeze(-1))
    return critic_loss


def compute_actor_loss(q_values):
    return -q_values.mean()


def run_ddpg(cfg, logger, trial=None):
    # 1)  Build the  logger
    best_reward = float("-inf")

    # 2) Create the environment agent
    train_env_agent, eval_env_agent = get_env_agents(cfg)

    # 3) Create the DDPG Agent
    (
        train_agent,
        eval_agent,
        actor,
        critic,
        # target_actor,
        target_critic,
    ) = create_ddpg_agent(cfg, train_env_agent, eval_env_agent)
    ag_actor = TemporalAgent(actor)
    # ag_target_actor = TemporalAgent(target_actor)
    q_agent = TemporalAgent(critic)
    target_q_agent = TemporalAgent(target_critic)

    train_workspace = Workspace()
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # Configure the optimizer
    actor_optimizer, critic_optimizer = setup_optimizers(cfg, actor, critic)
    nb_steps = 0
    tmp_steps = 0
    if cfg.collect_stats:
        directory = "./ddpg_data/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + "ddpg.data"
        fo = open(filename, "wb")
        stats_data = []

    # Training loop
    while nb_steps < cfg.algorithm.n_steps:
        # Execute the agent in the workspace
        if nb_steps > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(train_workspace, t=1, n_steps=cfg.algorithm.n_steps_train)
        else:
            train_agent(train_workspace, t=0, n_steps=cfg.algorithm.n_steps_train)

        transition_workspace = train_workspace.get_transitions(filter_key="env/done")
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]
        rb.put(transition_workspace)

        for _ in range(cfg.algorithm.optim_n_updates):
            rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

            terminated, reward, action = rb_workspace[
                "env/terminated", "env/reward", "action"
            ]
            if nb_steps > cfg.algorithm.learning_starts:
                # Determines whether values of the critic should be propagated
                # True if the task was not terminated.
                must_bootstrap = ~terminated[1]

                # Critic update
                # compute q_values: at t, we have Q(s,a) from the (s,a) in the RB
                # the detach_actions=True changes nothing in the results
                q_agent(rb_workspace, t=0, n_steps=1, detach_actions=True)
                q_values = rb_workspace["critic/q_values"]

                with torch.no_grad():
                    # replace the action at t+1 in the RB with \pi(s_{t+1}), to compute Q(s_{t+1}, \pi(s_{t+1}) below
                    ag_actor(rb_workspace, t=1, n_steps=1)
                    # compute q_values: at t+1 we have Q(s_{t+1}, \pi(s_{t+1})
                    target_q_agent(rb_workspace, t=1, n_steps=1, detach_actions=True)
                    # q_agent(rb_workspace, t=1, n_steps=1)
                # finally q_values contains the above collection at t=0 and t=1
                post_q_values = rb_workspace["target-critic/q_values"]

                # Compute critic loss
                critic_loss = compute_critic_loss(
                    cfg, reward, must_bootstrap, q_values[0], post_q_values[1]
                )
                logger.add_log("critic_loss", critic_loss, nb_steps)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), cfg.algorithm.max_grad_norm
                )
                critic_optimizer.step()

                # Actor update
                # Now we determine the actions the current policy would take in the states from the RB
                ag_actor(rb_workspace, t=0, n_steps=1)
                # We determine the Q values resulting from actions of the current policy
                q_agent(rb_workspace, t=0, n_steps=1)
                # and we back-propagate the corresponding loss to maximize the Q values
                q_values = rb_workspace["critic/q_values"]
                actor_loss = compute_actor_loss(q_values)
                logger.add_log("actor_loss", actor_loss, nb_steps)
                # if -25 < actor_loss < 0 and nb_steps > 2e5:
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), cfg.algorithm.max_grad_norm
                )
                actor_optimizer.step()
                # Soft update of target q function
                tau = cfg.algorithm.tau_target
                soft_update_params(critic, target_critic, tau)
                # soft_update_params(actor, target_actor, tau)

        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(eval_workspace, t=0, stop_variable="env/done")

            all_rewards = eval_workspace["env/cumulated_reward"]
            rewards = all_rewards[-1]

            mean = rewards.mean()
            logger.log_reward_losses(rewards, nb_steps)

            if mean > best_reward:
                best_reward = mean

            print(f"nb_steps: {nb_steps}, reward: {mean:.0f}, best: {best_reward:.0f}")

            if cfg.save_best and best_reward == mean:
                save_best(
                    eval_agent,
                    cfg.gym_env.env_name,
                    mean,
                    "./ddpg_best_agents/",
                    "ddpg",
                )
                if cfg.plot_agents:
                    plot_policy(
                        eval_agent.agent.agents[1],
                        eval_env_agent,
                        best_reward,
                        "./ddpg_plots/",
                        cfg.gym_env.env_name,
                        stochastic=False,
                    )
                    plot_critic(
                        critic,
                        eval_env_agent,
                        best_reward,
                        "./ddpg_plots/",
                        cfg.gym_env.env_name,
                    )

            if cfg.collect_stats:
                stats_data.append(rewards)

    if cfg.collect_stats:
        # All rewards, dimensions (# of evaluations x # of episodes)
        stats_data = torch.stack(stats_data, axis=-1)
        print(np.shape(stats_data))
        np.savetxt(filename, stats_data.numpy())
        fo.flush()
        fo.close()

    return best_reward


# %%
@hydra.main(
    config_path="configs/",
    # config_name="ddpg_cartpole.yaml"
    # config_name="ddpg_pendulum.yaml",
    config_name="ddpg_pendulum_optuna.yaml",
    # config_name="ddpg_pendulum_optimise.yaml",
)  # , version_base="1.3")
def main(cfg_raw: DictConfig):
    torch.random.manual_seed(seed=cfg_raw.algorithm.seed.torch)

    if "optuna" in cfg_raw:
        launch_optuna(cfg_raw, run_ddpg)
    else:
        logger = Logger(cfg_raw)
        run_ddpg(cfg_raw, logger)


if __name__ == "__main__":
    main()
