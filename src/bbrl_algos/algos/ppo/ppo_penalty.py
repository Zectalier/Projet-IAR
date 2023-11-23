"""
This version of PPO works, but it incorrectly samples minibatches randomly from the rollouts
without making sure that each sample is used once and only once
See: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
for a full description of all the coding tricks that should be integrated
"""


import sys
import os
import copy
import numpy as np

import torch
import torch.nn as nn

import optuna
import hydra

# from tqdm.auto import tqdm

from omegaconf import DictConfig

from bbrl import get_arguments, get_class

from bbrl.utils.functional import gae

from bbrl_algos.models.loggers import Logger
from bbrl.utils.chrono import Chrono

# The workspace is the main class in BBRL, this is where all data is collected and stored
from bbrl.workspace import Workspace

# Agents(agent1,agent2,agent3,...) executes the different agents the one after the other
# TemporalAgent(agent) executes an agent over multiple timesteps in the workspace,
# or until a given condition is reached
from bbrl.agents import Agents, TemporalAgent

# AutoResetGymAgent is an agent able to execute a batch of gym environments
# with auto-resetting. These agents produce multiple variables in the workspace:
# ’env/env_obs’, ’env/reward’, ’env/timestep’, ’env/done’, ’env/initial_state’, ’env/cumulated_reward’,
# ... When called at timestep t=0, then the environments are automatically reset.
# At timestep t>0, these agents will read the ’action’ variable in the workspace at time t − 1
from bbrl_algos.models.envs import get_env_agents
from bbrl_algos.models.hyper_params import launch_optuna
from bbrl_algos.models.utils import save_best

# Neural network models for policys and critics
from bbrl_algos.models.stochastic_actors import (
    TunableVariancePPOActor,
    TunableVarianceContinuousActor,
    TunableVarianceContinuousActorExp,
    SquashedGaussianActor,
    StateDependentVarianceContinuousActor,
    ConstantVarianceContinuousActor,
    DiscretePPOActor,
    BernoulliActor,
)
from bbrl_algos.models.critics import VAgent

# The KLAgent is specific to the KL regularization version of PPO
# It is used to compute the KL divergence between the current and the past policy
from bbrl_algos.models.exploration_agents import KLAgent

# Used to display a policy and a critic as a 2D map
from bbrl.visu.plot_policies import plot_policy
from bbrl.visu.plot_critics import plot_critic

import matplotlib

matplotlib.use("TkAgg")


# Create the PPO Agent
def create_ppo_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    policy = globals()[cfg.algorithm.policy_type](
        obs_size,
        cfg.algorithm.architecture.policy_hidden_size,
        act_size,
        name="current_policy",
        seed=cfg.algorithm.seed.policy,
    )
    tr_agent = Agents(train_env_agent, policy)
    ev_agent = Agents(eval_env_agent, policy)

    critic_agent = VAgent(obs_size, cfg.algorithm.architecture.critic_hidden_size)
    old_critic_agent = copy.deepcopy(critic_agent).set_name("old_critic")

    all_critics = TemporalAgent(Agents(critic_agent, old_critic_agent))

    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)

    old_policy = copy.deepcopy(policy)
    old_policy.set_name("old_policy")

    kl_agent = TemporalAgent(KLAgent(old_policy, policy))
    return (
        train_agent,
        eval_agent,
        critic_agent,
        all_critics,
        policy,
        old_policy,
        kl_agent,
    )


def setup_optimizer(cfg, policy, critic):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = nn.Sequential(policy, critic).parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer


def compute_advantage(cfg, reward, must_bootstrap, v_value):
    # Compute temporal difference with GAE
    reward = reward[1]
    next_val = v_value[1]
    mb = must_bootstrap[1]
    current_val = v_value[0]
    advantage = gae(
        reward,
        next_val,
        mb,
        current_val,
        cfg.algorithm.discount_factor,
        cfg.algorithm.gae,
    )
    return advantage


def compute_critic_loss(advantage):
    td_error = advantage**2
    critic_loss = td_error.mean()
    return critic_loss


def compute_penalty_policy_loss(cfg, advantage, ratio, kl_loss):
    """Computes the PPO loss including KL regularization"""
    policy_loss = (advantage * ratio - cfg.algorithm.beta * kl_loss).mean()
    return policy_loss


def run_ppo_penalty(cfg, logger, trial=None):
    best_reward = float("-inf")
    nb_steps = 0
    tmp_steps = 0
    if cfg.collect_stats:
        directory = "./ppo_data/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + "ppo_" + cfg.gym_env.env_name + ".data"
        fo = open(filename, "wb")
        stats_data = []

    train_env_agent, eval_env_agent = get_env_agents(cfg)

    (
        train_agent,
        eval_agent,
        critic_agent,
        all_critics,
        policy,
        old_policy_params,
        kl_agent,
    ) = create_ppo_agent(cfg, train_env_agent, eval_env_agent)

    # The old_policy params must be wrapped into a TemporalAgent
    old_policy = TemporalAgent(old_policy_params)

    train_workspace = Workspace()

    # Configure the optimizer
    optimizer = setup_optimizer(cfg, train_agent, critic_agent)

    # Training loop
    while nb_steps < cfg.algorithm.n_steps:
        # Execute the training agent in the workspace

        # Handles continuation
        delta_t = 0
        if nb_steps > 0:
            train_workspace.zero_grad()
            delta_t = 1
            train_workspace.copy_n_last_steps(1)

        # Run the current policy and evaluate the proba of its action according to the old policy
        # The old_policy can be run after the train_agent on the same workspace
        # because it writes a logprob_predict and not an action.
        # That is, it does not determine the action of the old_policy,
        # it just determines the proba of the action of the current policy given its own probabilities

        with torch.no_grad():
            train_agent(
                train_workspace,
                t=delta_t,
                n_steps=cfg.algorithm.n_steps_train,
                stochastic=True,
                predict_proba=False,
                compute_entropy=False,
            )
            old_policy(
                train_workspace,
                t=delta_t,
                n_steps=cfg.algorithm.n_steps_train,
                # Just computes the probability of the old policy's action
                # to get the ratio of probabilities
                predict_proba=True,
                compute_entropy=False,
            )

        # Compute the critic value over the whole workspace
        all_critics(train_workspace, t=delta_t, n_steps=cfg.algorithm.n_steps_train)

        transition_workspace = train_workspace.get_transitions()

        terminated, reward, action, v_value, old_v_value = transition_workspace[
            "env/terminated",
            "env/reward",
            "action",
            "critic/v_values",
            "old_critic/v_values",
        ]
        nb_steps += action[0].shape[0]

        assert (
            old_v_value.shape == v_value.shape
        ), f"{old_v_value.shape}[{old_v_value}]/{v_value.shape}[{v_value}]"
        # assert old_v_value.equal(v_value), f"{old_v_value.shape}[{old_v_value}]/{v_value.shape}[{v_value}]"
        if cfg.algorithm.clip_range_vf > 0:
            # Clip the difference between old and new values
            # NOTE: this depends on the reward scaling
            v_value = old_v_value + torch.clamp(
                v_value - old_v_value,
                -cfg.algorithm.clip_range_vf,
                cfg.algorithm.clip_range_vf,
            )

        # then we compute the advantage using the clamped critic values
        advantage = compute_advantage(cfg, reward, ~terminated, v_value)

        # We store the advantage into the transition_workspace
        transition_workspace.set("advantage", 0, advantage)

        critic_loss = compute_critic_loss(advantage)
        loss_critic = cfg.algorithm.critic_coef * critic_loss

        optimizer.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(
            critic_agent.parameters(), cfg.algorithm.max_grad_norm
        )
        optimizer.step()

        # We start several optimization epochs on mini_batches
        for opt_epoch in range(cfg.algorithm.opt_epochs):
            if cfg.algorithm.batch_size > 0:
                sample_workspace = transition_workspace.select_batch_n(
                    cfg.algorithm.batch_size
                )
            else:
                sample_workspace = transition_workspace

            kl_agent(sample_workspace, t=0, n_steps=1)
            kl = sample_workspace["kl"][0]

            # Compute the probability of the played actions according to the current policy
            # We do not replay the action: we use the one stored into the dataset
            # Hence predict_proba=True
            policy(
                sample_workspace,
                t=0,
                compute_entropy=True,
                predict_proba=True,
            )

            # The logprob_predict Tensor has been computed from the old_policy outside the loop
            advantage, action_logp, old_action_logp, entropy = sample_workspace[
                "advantage",
                "current_policy/logprob_predict",
                "old_policy/logprob_predict",
                "entropy",
            ]

            act_diff = action_logp[0] - old_action_logp[0].detach()
            ratios = act_diff.exp()

            policy_advantage = advantage.detach()[0]
            policy_loss = compute_penalty_policy_loss(cfg, policy_advantage, ratios, kl)
            loss_policy = -cfg.algorithm.policy_coef * policy_loss

            # Entropy loss favors exploration
            entropy_loss = entropy[0].mean()
            loss_entropy = -cfg.algorithm.entropy_coef * entropy_loss

            # Store the losses for tensorboard display
            logger.log_losses(critic_loss, entropy_loss, policy_loss, nb_steps)
            logger.add_log("advantage", policy_advantage.mean(), nb_steps)

            loss = loss_policy + loss_entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), cfg.algorithm.max_grad_norm
            )
            optimizer.step()

        old_policy_params.copy_parameters(policy)
        all_critics.agent.agents[1] = copy.deepcopy(critic_agent).set_name("old_critic")

        # Evaluate if enough steps have been performed
        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace,
                t=0,
                stop_variable="env/done",
                stochastic=False,
                predict_proba=False,
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()

            logger.log_reward_losses(rewards, nb_steps)
            print(
                f"nb_steps: {nb_steps}, reward: {mean:.3f}, best_reward: {best_reward:.3f}"
            )
            if mean > best_reward:
                best_reward = mean

            if cfg.save_best and best_reward == mean:
                save_best(
                    policy, cfg.gym_env.env_name, mean, "./ppo_best_agents/", "ppo"
                )
                if cfg.plot_agents:
                    plot_policy(
                        eval_agent.agent.agents[1],
                        eval_env_agent,
                        best_reward,
                        "./ppo_plots/",
                        cfg.gym_env.env_name,
                        stochastic=False,
                    )
                    plot_critic(
                        critic_agent,
                        eval_env_agent,
                        best_reward,
                        "./ppo_plots/",
                        cfg.gym_env.env_name,
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
        np.savetxt(filename, stats_data.numpy())
        fo.flush()
        fo.close()

    return best_reward


@hydra.main(
    config_path="./configs/",
    # config_name="ppo_lunarlander_continuous.yaml",
    # config_name="ppo_cartpole.yaml",
    config_name="ppo_lunarlander.yaml",
    # config_name="ppo_swimmer.yaml",
    # config_name="ppo_pendulum.yaml",
    # config_name="ppo_cartpole_continuous.yaml",
    # version_base="1.1",
)
def main(cfg_raw: DictConfig):
    torch.random.manual_seed(seed=cfg_raw.algorithm.seed.torch)
    chrono = Chrono()

    if "optuna" in cfg_raw:
        launch_optuna(cfg_raw, run_ppo_penalty)
    else:
        logger = Logger(cfg_raw)
        run_ppo_penalty(cfg_raw, logger)
    chrono.stop()


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
