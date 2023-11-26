import sys
import os
import copy
import torch
import torch.nn as nn
import hydra
import numpy as np
import optuna

from omegaconf import DictConfig
from bbrl.utils.chrono import Chrono

from bbrl import get_arguments, get_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from bbrl_algos.models.loggers import Logger
from bbrl.utils.replay_buffer import ReplayBuffer

from bbrl_algos.models.stochastic_actors import (
    SquashedGaussianActorNew,
    TunableVarianceContinuousActor,
    DiscreteActor,
)
from bbrl_algos.models.critics import ContinuousQAgent
from bbrl_algos.models.shared_models import soft_update_params
from bbrl_algos.models.envs import get_env_agents
from bbrl_algos.models.hyper_params import launch_optuna
from bbrl_algos.models.utils import save_best

from bbrl.visu.plot_policies import plot_policy
from bbrl.visu.plot_critics import plot_critic

import matplotlib

# HYDRA_FULL_ERROR = 1


matplotlib.use("TkAgg")


# Create the SAC Agent
def create_sac_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    assert (
        train_env_agent.is_continuous_action()
    ), "SAC code dedicated to continuous actions"
    actor = SquashedGaussianActorNew(
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size, name="policy"
    )
    tr_agent = Agents(train_env_agent, actor)
    ev_agent = Agents(eval_env_agent, actor)
    critic_1 = ContinuousQAgent(
        obs_size,
        cfg.algorithm.architecture.critic_hidden_size,
        act_size,
        name="critic-1",
    )
    target_critic_1 = copy.deepcopy(critic_1).set_name("target-critic-1")
    critic_2 = ContinuousQAgent(
        obs_size,
        cfg.algorithm.architecture.critic_hidden_size,
        act_size,
        name="critic-2",
    )
    target_critic_2 = copy.deepcopy(critic_2).set_name("target-critic-2")
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    return (
        train_agent,
        eval_agent,
        actor,
        critic_1,
        target_critic_1,
        critic_2,
        target_critic_2,
    )


# Configure the optimizer
def setup_optimizers(cfg, actor, critic_1, critic_2):
    actor_optimizer_args = get_arguments(cfg.actor_optimizer)
    parameters = actor.parameters()
    actor_optimizer = get_class(cfg.actor_optimizer)(parameters, **actor_optimizer_args)
    critic_optimizer_args = get_arguments(cfg.critic_optimizer)
    parameters = nn.Sequential(critic_1, critic_2).parameters()
    critic_optimizer = get_class(cfg.critic_optimizer)(
        parameters, **critic_optimizer_args
    )
    return actor_optimizer, critic_optimizer


def setup_entropy_optimizers(cfg):
    if cfg.algorithm.entropy_mode == "auto":
        entropy_coef_optimizer_args = get_arguments(cfg.entropy_coef_optimizer)
        # Note: we optimize the log of the entropy coef which is slightly different from the paper
        # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
        # Comment and code taken from the SB3 version of SAC
        log_entropy_coef = torch.log(
            torch.ones(1) * cfg.algorithm.init_entropy_coef
        ).requires_grad_(True)
        entropy_coef_optimizer = get_class(cfg.entropy_coef_optimizer)(
            [log_entropy_coef], **entropy_coef_optimizer_args
        )
        return entropy_coef_optimizer, log_entropy_coef
    else:
        return None, None


# %%
def compute_critic_loss(
    cfg,
    reward,
    must_bootstrap,
    current_actor,
    q_agents,
    target_q_agents,
    rb_workspace,
    ent_coef,
):
    """
    Computes the critic loss for a set of $S$ transition samples

    Args:
        cfg: The experimental configuration
        reward: Tensor (2xS) of rewards
        must_bootstrap: Tensor (S) of indicators
        current_actor: The actor agent (as a TemporalAgent)
        q_agents: The critics (as a TemporalAgent)
        target_q_agents: The target of the critics (as a TemporalAgent)
        rb_workspace: The transition workspace
        ent_coef: The entropy coefficient

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The two critic losses (scalars)
    """

    # Compute q_values from both critics with the actions present in the buffer:
    # at t, we have Q(s,a) from the (s,a) in the RB
    q_agents(rb_workspace, t=0, n_steps=1)

    with torch.no_grad():
        # Replay the current actor on the replay buffer to get actions of the
        # current actor
        current_actor(rb_workspace, t=1, n_steps=1, stochastic=True)

        # Compute target q_values from both target critics: at t+1, we have
        # Q(s+1,a+1) from the (s+1,a+1) where a+1 has been replaced in the RB
        target_q_agents(rb_workspace, t=1, n_steps=1)

        action_logprobs_next = rb_workspace["policy/action_logprobs"]

    q_values_rb_1, q_values_rb_2, post_q_values_1, post_q_values_2 = rb_workspace[
        "critic-1/q_values",
        "critic-2/q_values",
        "target-critic-1/q_values",
        "target-critic-2/q_values",
    ]

    # [[student]] Compute temporal difference

    q_next = torch.min(post_q_values_1[1], post_q_values_2[1]).squeeze(-1)
    v_phi = q_next - ent_coef * action_logprobs_next[1]

    target = reward[-1] + cfg.algorithm.discount_factor * v_phi * must_bootstrap.int()
    td_1 = target - q_values_rb_1[0].squeeze(-1)
    td_2 = target - q_values_rb_2[0].squeeze(-1)
    td_error_1 = td_1**2
    td_error_2 = td_2**2
    critic_loss_1 = td_error_1.mean()
    critic_loss_2 = td_error_2.mean()
    # [[/student]]

    return critic_loss_1, critic_loss_2


# %%
def compute_actor_loss(ent_coef, current_actor, q_agents, rb_workspace):
    """
    Actor loss computation
    :param ent_coef: The entropy coefficient $\alpha$
    :param current_actor: The actor agent (temporal agent)
    :param q_agents: The critics (as temporal agent)
    :param rb_workspace: The replay buffer (2 time steps, $t$ and $t+1$)
    """

    # Recompute the q_values from the current actor, not from the actions in the buffer

    current_actor(rb_workspace, t=0, n_steps=1, stochastic=True)
    action_logprobs_new = rb_workspace["policy/action_logprobs"]

    q_agents(rb_workspace, t=0, n_steps=1)
    q_values_1, q_values_2 = rb_workspace["critic-1/q_values", "critic-2/q_values"]

    current_q_values = torch.min(q_values_1, q_values_2).squeeze(-1)

    actor_loss = ent_coef * action_logprobs_new[0] - current_q_values[0]

    return actor_loss.mean()


def run_sac(cfg, logger, trial=None):
    best_reward = float("-inf")

    # init_entropy_coef is the initial value of the entropy coef alpha.
    ent_coef = cfg.algorithm.init_entropy_coef
    tau = cfg.algorithm.tau_target

    # 2) Create the environment agent
    train_env_agent, eval_env_agent = get_env_agents(cfg)

    # 3) Create the SAC Agent
    (
        train_agent,
        eval_agent,
        actor,
        critic_1,
        target_critic_1,
        critic_2,
        target_critic_2,
    ) = create_sac_agent(cfg, train_env_agent, eval_env_agent)

    current_actor = TemporalAgent(actor)
    q_agents = TemporalAgent(Agents(critic_1, critic_2))
    target_q_agents = TemporalAgent(Agents(target_critic_1, target_critic_2))
    train_workspace = Workspace()

    # Creates a replay buffer
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # Configure the optimizer
    actor_optimizer, critic_optimizer = setup_optimizers(cfg, actor, critic_1, critic_2)
    entropy_coef_optimizer, log_entropy_coef = setup_entropy_optimizers(cfg)
    nb_steps = 0
    tmp_steps = 0

    # If entropy_mode is not auto, the entropy coefficient ent_coef will remain fixed
    if cfg.algorithm.entropy_mode == "auto":
        # target_entropy is \mathcal{H}_0 in the SAC and aplications paper.
        target_entropy = -np.prod(train_env_agent.action_space.shape).astype(np.float32)

    # Training loop
    while nb_steps < cfg.algorithm.n_steps:
        # Execute the agent in the workspace
        if nb_steps > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(
                train_workspace,
                t=1,
                n_steps=cfg.algorithm.n_steps_train,
                stochastic=True,
            )
        else:
            train_agent(
                train_workspace,
                t=0,
                n_steps=cfg.algorithm.n_steps_train,
                stochastic=True,
            )

        transition_workspace = train_workspace.get_transitions()
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]
        rb.put(transition_workspace)

        if nb_steps > cfg.algorithm.learning_starts:
            # Get a sample from the workspace
            rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

            terminated, reward = rb_workspace["env/terminated", "env/reward"]
            if entropy_coef_optimizer is not None:
                ent_coef = torch.exp(log_entropy_coef.detach())

            # Critic update part #
            critic_optimizer.zero_grad()

            (critic_loss_1, critic_loss_2) = compute_critic_loss(
                cfg,
                reward,
                ~terminated[1],
                current_actor,
                q_agents,
                target_q_agents,
                rb_workspace,
                ent_coef,
            )

            logger.add_log("critic_loss_1", critic_loss_1, nb_steps)
            logger.add_log("critic_loss_2", critic_loss_2, nb_steps)
            critic_loss = critic_loss_1 + critic_loss_2
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                critic_1.parameters(), cfg.algorithm.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                critic_2.parameters(), cfg.algorithm.max_grad_norm
            )
            critic_optimizer.step()

            # Actor update part #
            actor_optimizer.zero_grad()
            actor_loss = compute_actor_loss(
                ent_coef, current_actor, q_agents, rb_workspace
            )
            logger.add_log("actor_loss", actor_loss, nb_steps)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), cfg.algorithm.max_grad_norm
            )
            actor_optimizer.step()

            # Entropy coef update part #
            if entropy_coef_optimizer is not None:
                # See Eq. (17) of the SAC and Applications paper
                # log. probs have been computed when computing the actor loss
                action_logprobs_rb = rb_workspace["policy/action_logprobs"].detach()
                entropy_coef_loss = -(
                    log_entropy_coef.exp() * (action_logprobs_rb + target_entropy)
                ).mean()
                entropy_coef_optimizer.zero_grad()
                entropy_coef_loss.backward()
                entropy_coef_optimizer.step()
                logger.add_log("entropy_coef_loss", entropy_coef_loss, nb_steps)
            logger.add_log("entropy_coef", ent_coef, nb_steps)

            # Soft update of target q function
            soft_update_params(critic_1, target_critic_1, tau)
            soft_update_params(critic_2, target_critic_2, tau)
            # soft_update_params(actor, target_actor, tau)

        # Evaluate
        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace,
                t=0,
                stop_variable="env/done",
                stochastic=False,
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.log_reward_losses(rewards, nb_steps)

            if mean > best_reward:
                best_reward = mean

            print(
                f"nb steps: {nb_steps}, reward: {mean:.02f}, best: {best_reward:.02f}"
            )
            if cfg.save_best and best_reward == mean:
                save_best(
                    actor, cfg.gym_env.env_name, mean, "./sac_best_agents/", "sac"
                )

    return best_reward


def load_best(best_filename):
    best_agent = torch.load(best_filename)
    return best_agent


# %%
@hydra.main(
    config_path="./configs/",
    # config_name="sac_lunar_lander_continuous.yaml",
    # config_name="sac_walker_optuna.yaml",
    config_name="sac_hopper_optuna.yaml",
    # config_name="sac_cartpolecontinuous.yaml",
    # config_name="sac_pendulum.yaml",
    # config_name="sac_swimmer_optuna.yaml",
    # config_name="sac_swimmer.yaml",
    # config_name="sac_torcs.yaml",
    # version_base="1.3",
)
def main(cfg_raw: DictConfig):
    torch.random.manual_seed(seed=cfg_raw.algorithm.seed.torch)

    if "optuna" in cfg_raw:
        launch_optuna(cfg_raw, run_sac)
    else:
        logger = Logger(cfg_raw)
        run_sac(cfg_raw, logger)


if __name__ == "__main__":
    main()
