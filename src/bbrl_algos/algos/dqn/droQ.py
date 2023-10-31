#
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

import copy
import os
import numpy as np
from typing import Callable, List

import hydra
import optuna
from omegaconf import DictConfig

# %%
import torch
import torch.nn as nn

# %%
from bbrl.agents.gymnasium import record_video
from moviepy.editor import ipython_display as video_display
import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import AutoResetWrapper

# %%
from bbrl import get_arguments, get_class
from bbrl.agents import TemporalAgent, Agents, PrintAgent
from bbrl.workspace import Workspace

from bbrl_algos.models.exploration_agents import EGreedyActionSelector
from bbrl_algos.models.loggers import Logger
from bbrl_algos.models.utils import save_best

from bbrl.visu.plot_critics import plot_discrete_q, plot_critic
from bbrl_algos.models.hyper_params import launch_optuna

from bbrl.utils.functional import gae
from bbrl.utils.chrono import Chrono

# HYDRA_FULL_ERROR = 1
import matplotlib
import matplotlib.pyplot as plt

from bbrl_gymnasium.envs.maze_mdp import MazeMDPEnv
from bbrl_algos.wrappers.env_wrappers import MazeMDPContinuousWrapper
from bbrl.agents.gymnasium import make_env, ParallelGymAgent
from functools import partial

from bbrl.utils.replay_buffer import ReplayBuffer

import sys
sys.path.append("../../")
from models.dropout_critics import DiscreteDropoutQAgent

matplotlib.use("Agg")


def local_get_env_agents(cfg):
    eval_env_agent = ParallelGymAgent(
        partial(
            make_env,
            cfg.gym_env.env_name,
            autoreset=False,
        ),
        cfg.algorithm.nb_evals,
        include_last_state=True,
        seed=cfg.algorithm.seed.eval,
    )
    train_env_agent = ParallelGymAgent(
        partial(
            make_env,
            cfg.gym_env.env_name,
            autoreset=True,
        ),
        cfg.algorithm.n_envs,
        include_last_state=True,
        seed=cfg.algorithm.seed.train,
    )
    return train_env_agent, eval_env_agent


# %%
def compute_critic_loss(
    discount_factor, reward, must_bootstrap, action, q_values, q_target=None
):
    """Compute critic loss
    Args:
        discount_factor (float): The discount factor
        reward (torch.Tensor): a (2 × T × B) tensor containing the rewards
        must_bootstrap (torch.Tensor): a (2 × T × B) tensor containing 0 if the episode is completed at time $t$
        action (torch.LongTensor): a (2 × T) long tensor containing the chosen action
        q_values (torch.Tensor): a (2 × T × B × A) tensor containing Q values
        q_target (torch.Tensor, optional): a (2 × T × B × A) tensor containing target Q values

    Returns:
        torch.Scalar: The loss
    """
    
    with torch.no_grad():
        # Select the action according to the current Q-network
        current_q_values = q_values[1]
        selected_actions = current_q_values.argmax(dim=-1)
        
        # Estimate the Q-value of the selected action using the target Q-network
        target_q_values = q_target[1].gather(dim=-1, index=selected_actions.unsqueeze(-1)).squeeze(-1)
        target = reward[1] + discount_factor * target_q_values * must_bootstrap[1]

    # Get the Q-value of the action taken from the current Q-network
    qvals = q_values[0].gather(dim=1, index=action[0].unsqueeze(dim=-1)).squeeze(dim=1)

    # Compute the loss
    return nn.MSELoss()(qvals, target)



# %%
def create_dqn_agent(cfg_algo, train_env_agent, eval_env_agent):
    # obs_space = train_env_agent.get_observation_space()
    # obs_shape = obs_space.shape if len(obs_space.shape) > 0 else obs_space.n

    # act_space = train_env_agent.get_action_space()
    # act_shape = act_space.shape if len(act_space.shape) > 0 else act_space.n

    state_dim, action_dim = train_env_agent.get_obs_and_actions_sizes() 

    critic = DiscreteDropoutQAgent(
        state_dim=state_dim,
        hidden_dim=list([cfg_algo.architecture.hidden_size_1, cfg_algo.architecture.hidden_size_2]),
        action_dim=action_dim,
        seed=cfg_algo.seed.q,
        dropout=cfg_algo.architecture.dropout,
    )
    target_critic = copy.deepcopy(critic)

    explorer = EGreedyActionSelector(
        name="action_selector",
        epsilon=cfg_algo.explorer.epsilon_start,
        epsilon_end=cfg_algo.explorer.epsilon_end,
        epsilon_decay=cfg_algo.explorer.decay,
        seed=cfg_algo.seed.explorer,
    )
    q_agent = TemporalAgent(critic)
    target_q_agent = TemporalAgent(target_critic)

    tr_agent = Agents(train_env_agent, critic, explorer)  # , PrintAgent())
    ev_agent = Agents(eval_env_agent, critic)

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    

    return train_agent, eval_agent, q_agent, target_q_agent


# %%
# Configure the optimizer over the q agent
def setup_optimizer(optimizer_cfg, q_agent):
    optimizer_args = get_arguments(optimizer_cfg)
    parameters = q_agent.parameters()
    optimizer = get_class(optimizer_cfg)(parameters, **optimizer_args)
    return optimizer


# %%
def run_dqn(cfg, logger, trial=None):
    best_reward = float("-inf")
    best_agent = None

    if cfg.collect_stats:
        directory = "./droQ_data/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + "droQ_" + cfg.gym_env.env_name + ".data"
        fo = open(filename, "wb")
        stats_data = []

    # 1) Create the environment agent
    train_env_agent, eval_env_agent = local_get_env_agents(cfg)
    print(train_env_agent.envs[0])
    print(eval_env_agent.envs[0])

    # 2) Create the DQN-like Agent
    train_agent, eval_agent, q_agent, target_q_agent = create_dqn_agent(
        cfg.algorithm, train_env_agent, eval_env_agent
    )

    # 3) Create the training workspace
    train_workspace = Workspace()  # Used for training
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer.buffer_size)

    # 5) Configure the optimizer
    optimizer = setup_optimizer(cfg.optimizer, q_agent)

    # 6) Define the steps counters
    nb_steps = 0
    tmp_steps_eval = 0
    last_critic_update_step = 0

    while nb_steps < cfg.algorithm.n_steps:
        # Decay the explorer epsilon
        explorer = train_agent.agent.get_by_name("action_selector")
        assert len(explorer) == 1, "There should be only one explorer"
        explorer[0].decay()

        # Execute the agent in the workspace
        if nb_steps > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(
                train_workspace,
                t=1,
                n_steps=cfg.algorithm.n_steps_train - 1,
            )
        else:
            train_agent(
                train_workspace,
                t=0,
                n_steps=cfg.algorithm.n_steps_train,
            )

        # Get the transitions
        transition_workspace = train_workspace.get_transitions(filter_key="env/done")

        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]

        # Adds the transitions to the workspace
        rb.put(transition_workspace)

        if rb.size() > cfg.algorithm.buffer.learning_starts:
            for _ in range(cfg.algorithm.n_updates):
                rb_workspace = rb.get_shuffled(cfg.algorithm.buffer.batch_size)

                # The q agent needs to be executed on the rb_workspace workspace (gradients are removed in workspace).
                q_agent(rb_workspace, t=0, n_steps=2, choose_action=False)

                q_values, terminated, reward, action = rb_workspace[
                    "critic/q_values",
                    "env/terminated",
                    "env/reward",
                    "action",
                ]

                with torch.no_grad():
                            target_q_agent(rb_workspace, t=0, n_steps=2, stochastic=True)
                target_q_values = rb_workspace["critic/q_values"]

                # Determines whether values of the critic should be propagated
                # True if the task was not terminated.
                must_bootstrap = ~terminated

                critic_loss = compute_critic_loss(
                    cfg.algorithm.discount_factor,
                    reward,
                    must_bootstrap,
                    action,
                    q_values,
                    target_q_values
                )

                # Store the loss
                logger.add_log("critic_loss", critic_loss, nb_steps)

                optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    q_agent.parameters(), cfg.algorithm.max_grad_norm
                )

                optimizer.step()

                # Update the target network 
                if nb_steps - last_critic_update_step > cfg.algorithm.target_critic_update_interval:
                    last_critic_update_step = nb_steps
                    target_q_agent.agent = copy.deepcopy(q_agent.agent)

                # Evaluate the agent
                if nb_steps - tmp_steps_eval > cfg.algorithm.eval_interval:
                    tmp_steps_eval = nb_steps
                    eval_workspace = Workspace()  # Used for evaluation
                    eval_agent(
                        eval_workspace,
                        t=0,
                        stop_variable="env/done",
                        choose_action=True,
                    )
                    rewards = eval_workspace["env/cumulated_reward"][-1]
                    logger.log_reward_losses(rewards, nb_steps)
                    mean = rewards.mean()

                    if mean > best_reward:
                        best_reward = mean
                        best_agent = copy.deepcopy(eval_agent)

                    print(
                        f"nb_steps: {nb_steps}, reward: {mean:.02f}, best: {best_reward:.02f}"
                    )

                    if trial is not None:
                        trial.report(mean, nb_steps)
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                    if cfg.save_best and best_reward == mean:
                        save_best(
                            eval_agent,
                            cfg.gym_env.env_name,
                            best_reward,
                            "./droQ_best_agents/",
                            "droQ",
                        )
                        if cfg.plot_agents:
                            critic = eval_agent.agent.agents[1]
                            plot_discrete_q(
                                critic,
                                eval_env_agent,
                                best_reward,
                                "./droQ_plots/",
                                cfg.gym_env.env_name,
                                input_action="policy",
                            )
                            plot_discrete_q(
                                critic,
                                eval_env_agent,
                                best_reward,
                                "./droQ_plots2/",
                                cfg.gym_env.env_name,
                                input_action=None,
                            )
                    if cfg.collect_stats:
                        stats_data.append(rewards)

                    if trial is not None:
                        trial.report(mean, nb_steps)
                        if trial.should_prune():
                            raise optuna.TrialPruned()

    if cfg.collect_stats:
        # All rewards, dimensions (# of evaluations x # of episodes)
        stats_data = torch.stack(stats_data, axis=-1)
        print(np.shape(stats_data))
        np.savetxt(filename, stats_data.numpy(), fmt='%.4f', delimiter=' ')
        fo.flush()
        fo.close()
    
    return best_reward


# %%
@hydra.main(
    config_path="configs/",
    # config_name="dqn_cartpole.yaml",
    config_name="droQ_lunar_lander_optuna.yaml",
)  # , version_base="1.3")
def main(cfg_raw: DictConfig):
    torch.random.manual_seed(seed=cfg_raw.algorithm.seed.torch)

    if "optuna" in cfg_raw:
        launch_optuna(cfg_raw, run_dqn)
    else:
        logger = Logger(cfg_raw)
        run_dqn(cfg_raw, logger)

    print('End of training...')

    # 

if __name__ == "__main__":
    main()
