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
from bbrl.agents import Agents, TemporalAgent, Agent

from bbrl_algos.models.loggers import Logger
from bbrl.utils.replay_buffer import ReplayBuffer

from bbrl_algos.models.stochastic_actors import (
    SquashedGaussianActor,
    TunableVarianceContinuousActor,
    DiscreteActor,
)
from bbrl_algos.models.critics import ContinuousQAgent
from bbrl_algos.models.shared_models import soft_update_params

import sys
sys.path.append('/users/nfs/Etu7/21201287/Documents/bbrl_algos/src/')
from bbrl_algos.models.envs import get_env_agents

from bbrl_algos.models.hyper_params import launch_optuna
from bbrl_algos.models.utils import save_best

from bbrl.visu.plot_policies import plot_policy
from bbrl.visu.plot_critics import plot_critic

import matplotlib
import warnings

# HYDRA_FULL_ERROR = 1


matplotlib.use("Agg")


# Create the REDQ Agent TODO
def create_redq_agent(cfg, train_env_agent, eval_env_agent, M):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    
    assert (
        train_env_agent.is_continuous_action()
    ), "REDQ code dedicated to continuous actions"
    actor = SquashedGaussianActor(
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size, name="policy"
    )
    tr_agent = Agents(train_env_agent, actor)
    ev_agent = Agents(eval_env_agent, actor)

    # Create M critics
    critics = [
        ContinuousQAgent(
            obs_size,
            cfg.algorithm.architecture.critic_hidden_size,
            act_size,
            name=f"critic-{i+1}",
        )
        for i in range(M)
    ]
    target_critics = [
        copy.deepcopy(critics[i]).set_name(f"target-critic-{i+1}") for i in range(M)
    ]

    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)

    return (
        train_agent,
        eval_agent,
        actor,
        critics,
        target_critics,
    )


# Configure the optimizer
def setup_optimizers(cfg, actor, critics):
    actor_optimizer_args = get_arguments(cfg.actor_optimizer)
    parameters = actor.parameters()
    actor_optimizer = get_class(cfg.actor_optimizer)(parameters, **actor_optimizer_args)
    critic_optimizer_args = get_arguments(cfg.critic_optimizer)
    parameters = [param for critic in critics for param in critic.parameters()]
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
    M
):
    """
    Computes the critic loss for a set of $S$ transition samples and returns the M critic losses

    Args:
        cfg: The experimental configuration
        reward: Tensor (2xS) of rewards
        must_bootstrap: Tensor (S) of indicators
        current_actor: The actor agent (as a TemporalAgent)
        q_agents: The critics (as a TemporalAgent)
        target_q_agents: The target of the critics (as a TemporalAgent)
        rb_workspace: The transition workspace
        ent_coef: The entropy coefficient
        M: The number of critics

    Returns:
        Tuple[Tensor]: The critic losses for each critic
    """

    # Compute q_values from all the critics with the actions present in the buffer:
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

    # For each critic, get the q_values and post_q_values
    q_values_rb = []
    post_q_values = []
    for i in range(M):
        q_values_rb.append(rb_workspace[f"critic-{i+1}/q_values"])
        post_q_values.append(rb_workspace[f"target-critic-{i+1}/q_values"])

    # [[student]] Compute temporal difference error for each critic
    # Compute the q_next from the M post_q_values

    q_next = torch.min(torch.stack([q[1] for q in post_q_values]), dim=0)[0].squeeze(-1)
    v_phi = q_next - ent_coef * action_logprobs_next
    critic_losses = []
    
    target = reward[-1] + cfg.algorithm.discount_factor * v_phi * must_bootstrap.int()

    # Compute the critic loss for each critic
    for i in range(M):
        td = target - q_values_rb[i].squeeze(-1)
        td_error = td**2
        critic_losses.append(td_error.mean())
    # [[/student]]

    return critic_losses


# %%
def compute_actor_loss(ent_coef, current_actor, q_agents, rb_workspace, M):
    """
    Actor loss computation
    :param ent_coef: The entropy coefficient $\alpha$
    :param current_actor: The actor agent (temporal agent)
    :param q_agents: The critics (as temporal agent)
    :param rb_workspace: The replay buffer (2 time steps, $t$ and $t+1$)
    :param M: The number of critics
    """

    # Recompute the q_values from the current actor, not from the actions in the buffer

    current_actor(rb_workspace, t=0, n_steps=1, stochastic=True)
    action_logprobs_new = rb_workspace["policy/action_logprobs"]
    q_values = []

    q_agents(rb_workspace, t=0, n_steps=1)
    for i in range(M):
        q_values.append(rb_workspace[f"critic-{i+1}/q_values"])

    # Compute the q_values sum.
    current_q_values = sum(q_values) / M

    actor_loss = -current_q_values[0].squeeze(-1) - ent_coef * action_logprobs_new[1]
    
    return torch.mean(actor_loss)


def run_redq(cfg, logger, trial=None):
    best_reward = float("-inf")
    stats_data = []
    steps_data = []

    # init_entropy_coef is the initial value of the entropy coef alpha.
    ent_coef = cfg.algorithm.init_entropy_coef
    tau = cfg.algorithm.tau_target
    # Number of critics
    M = cfg.algorithm.M
    
    # 2) Create the environment agent
    train_env_agent, eval_env_agent = get_env_agents(cfg)

    # 3) Create the REDQ Agent
    (
        train_agent,
        eval_agent,
        actor,
        critics,
        target_critics,
    ) = create_redq_agent(cfg, train_env_agent, eval_env_agent, M)

    current_actor = TemporalAgent(actor)
    q_agents = TemporalAgent(Agents(*critics))
    target_q_agents = TemporalAgent(Agents(*target_critics))
    train_workspace = Workspace()

    # Creates a replay buffer
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # Configure the optimizer
    actor_optimizer, critic_optimizer = setup_optimizers(cfg, actor, critics)
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

        # For G Updates
        for updates in range(cfg.algorithm.utd_ratio):
            if nb_steps > cfg.algorithm.learning_starts:
                # Get a sample from the workspace
                rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

                terminated, reward = rb_workspace["env/terminated", "env/reward"]
                if entropy_coef_optimizer is not None:
                    ent_coef = torch.exp(log_entropy_coef.detach())

                # Critic update part #
                critic_optimizer.zero_grad()

                critic_losses = compute_critic_loss(
                    cfg,
                    reward,
                    ~terminated[1],
                    current_actor,
                    q_agents,
                    target_q_agents,
                    rb_workspace,
                    ent_coef,
                    M
                )

                for critic_loss in range(len(critic_losses)):
                    logger.add_log(f"critic_loss_{critic_loss}", critic_losses[critic_loss], nb_steps)

                critic_loss = sum(critic_losses)
                critic_loss.backward()

                for critic in critics:
                    torch.nn.utils.clip_grad_norm_(
                        critic.parameters(), cfg.algorithm.max_grad_norm
                    )

                critic_optimizer.step()

                # Actor update part #
                actor_optimizer.zero_grad()
                actor_loss = compute_actor_loss(
                    ent_coef, current_actor, q_agents, rb_workspace, M
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
                for critic, target_critic in zip(critics, target_critics):
                    soft_update_params(critic, target_critic, tau)
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
                        actor, cfg.gym_env.env_name, mean, "./redq_best_agents/", "redq"
                    )
                if cfg.collect_stats:
                    stats_data.append(rewards)
                    steps_data.append(nb_steps)

    if cfg.collect_stats:
        directory = cfg.stats_directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + "redq-" + cfg.gym_env.env_name + ".data"
        # Count the number of files with droQ-steps-*.data in the directory
        run_number = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and "droQ-steps-" in f])

        filename_steps = directory + "droQ-steps-" + str(run_number) + "-" + cfg.gym_env.env_name + ".data"
        # Append the stats_data to the file as a numpy array without overwriting

        # All rewards, dimensions (# of evaluations x # of episodes)
        stats_data = torch.stack(stats_data, axis=-1)

        # Only create the file if it does not exist.
        if not os.path.isfile(filename):
            fo = open(filename, "wb")
            fo.close()

        if not os.path.isfile(filename_steps):
            fo_steps = open(filename_steps, "wb")
            fo_steps.close()

        old_stats_data = np.array([])
        old_steps_data = np.array([])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                old_stats_data = np.loadtxt(filename)
                old_steps_data = np.loadtxt(filename_steps)
            except:
                pass
        
        if old_stats_data.shape != (0,):
            # Get the number of episodes in the old data
            old_n_episodes = old_stats_data.shape[1]
            # Get the number of episodes in the new data
            new_n_episodes = stats_data.shape[1]

            # Remove the extra episodes from the new data if there are more episodes in the new data
            if new_n_episodes > old_n_episodes:
                stats_data = stats_data[:, :old_n_episodes]
            # Remove the extra episodes from the old data if there are more episodes in the old data
            elif new_n_episodes < old_n_episodes:
                old_stats_data = old_stats_data[:, :new_n_episodes]
                
            # Concatenate the new rewards to the existing array
            new_stats_data = np.concatenate((old_stats_data, stats_data), axis=0)
            new_steps_data = np.concatenate((old_steps_data, steps_data), axis=0)
            
            fo = open(filename, "rb+")  # Open in read/write mode
            fo_steps = open(filename_steps, "rb+")
            np.savetxt(fo, new_stats_data, fmt='%.4f', delimiter=' ')
            np.savetxt(fo_steps, new_steps_data, fmt='%.4f', delimiter=' ')
        else:
            fo = open(filename, "wb")
            fo_steps = open(filename_steps, "wb")
            np.savetxt(fo, stats_data, fmt='%.4f', delimiter=' ')
            np.savetxt(fo_steps, steps_data, fmt='%.4f', delimiter=' ')

        fo.flush()
        fo.close()

    return best_reward


def load_best(best_filename):
    best_agent = torch.load(best_filename)
    return best_agent


# %%
@hydra.main(
    config_path="./configs/hopper/",
    #config_path="./configs/walker/",
    # config_name="redq_hopper_optuna.yaml",
    config_name="redq_hopper.yaml",
    # config_name="redq_walker_optuna.yaml",
    # config_name="redq_walker.yaml",
    # version_base="1.3",
)
def main(cfg_raw: DictConfig):
    torch.random.manual_seed(seed=cfg_raw.algorithm.seed.torch)

    if "optuna" in cfg_raw:
        launch_optuna(cfg_raw, run_redq)
    else:
        logger = Logger(cfg_raw)
        run_redq(cfg_raw, logger)


if __name__ == "__main__":
    main()
