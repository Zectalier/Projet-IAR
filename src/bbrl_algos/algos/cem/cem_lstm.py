import numpy as np

import torch
import torch.nn as nn
import hydra
import optuna

from omegaconf import DictConfig
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from bbrl_algos.models.loggers import Logger
from bbrl_algos.wrappers.env_wrappers import FilterWrapper
from bbrl_algos.models.hyper_params import launch_optuna
from bbrl_algos.models.utils import save_best
from bbrl_algos.models.envs import get_eval_env_agent

# Neural network models for actors and critics
from bbrl_algos.models.actors import (
    ContinuousDeterministicActor,
    DiscreteDeterministicActor,
)
from bbrl_algos.models.stochastic_actors import (
    TunableVariancePPOActor,
    TunableVarianceContinuousActor,
    TunableVarianceContinuousActorExp,
    SquashedGaussianActor,
    StateDependentVarianceContinuousActor,
    ConstantVarianceContinuousActor,
    DiscreteActor,
    BernoulliActor,
)
from bbrl_algos.models.envs import create_no_reset_env_agent

from bbrl.visu.plot_policies import plot_policy

import matplotlib

matplotlib.use("TkAgg")


class CovMatrix:
    def __init__(self, centroid: torch.Tensor, sigma, noise_multiplier):
        policy_dim = centroid.size()[0]
        self.noise = torch.diag(torch.ones(policy_dim) * sigma)
        self.cov = torch.diag(torch.ones(policy_dim) * torch.var(centroid)) + self.noise
        self.noise_multiplier = noise_multiplier

    def update_noise(self) -> None:
        self.noise = self.noise * self.noise_multiplier

    def generate_weights(self, centroid, pop_size):
        dist = torch.distributions.MultivariateNormal(
            centroid, covariance_matrix=self.cov
        )
        weights = [dist.sample() for _ in range(pop_size)]
        return weights

    def update_covariance(self, elite_weights) -> None:
        self.cov = torch.cov(elite_weights.T) + self.noise


# Create the CEM Agent
def create_CEM_agent(cfg, env_agent):
    obs_size, act_size = env_agent.get_obs_and_actions_sizes()
    policy = globals()[cfg.algorithm.actor_type](
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
    )
    ev_agent = Agents(env_agent, policy)
    eval_agent = TemporalAgent(ev_agent)

    return eval_agent


def run_cem(cfg, logger, trial=None):
    eval_env_agent = get_eval_env_agent(cfg)

    pop_size = cfg.algorithm.pop_size

    eval_agent = create_CEM_agent(cfg, eval_env_agent)

    centroid = torch.nn.utils.parameters_to_vector(eval_agent.parameters())
    matrix = CovMatrix(
        centroid,
        cfg.algorithm.sigma,
        cfg.algorithm.noise_multiplier,
    )

    best_score = -np.inf
    nb_steps = 0

    # 7) Training loop
    while nb_steps < cfg.algorithm.n_steps:
        matrix.update_noise()
        scores = []
        weights = matrix.generate_weights(centroid, pop_size)

        for i in range(pop_size):
            workspace = Workspace()
            w = weights[i]
            torch.nn.utils.vector_to_parameters(w, eval_agent.parameters())

            eval_agent(workspace, t=0, stop_variable="env/done", render=False)
            action = workspace["action"]
            nb_steps += action.shape[0]
            rewards = workspace["env/cumulated_reward"][-1]
            mean_reward = rewards.mean()
            logger.add_log("reward", mean_reward, nb_steps)

            # ---------------------------------------------------
            scores.append(mean_reward)

            if cfg.verbose:
                print(
                    f"Indiv: {i + 1}, nb_steps: {nb_steps}, reward: {mean_reward:.2f}"
                )
            if cfg.save_best and mean_reward > best_score:
                best_score = mean_reward
                print("Best score: ", best_score)
                save_best(
                    eval_agent,
                    cfg.gym_env.env_name,
                    mean_reward,
                    "./cem_best_agents/",
                    "cem",
                )
                if cfg.plot_agents:
                    plot_policy(
                        eval_agent.agent.agents[1],
                        eval_env_agent,
                        best_score,
                        "./cem_plots/",
                        cfg.gym_env.env_name,
                        stochastic=False,
                    )
        # Keep only best individuals to compute the new centroid
        elites_idxs = np.argsort(scores)[-cfg.algorithm.elites_nb :]
        elites_weights = [weights[k] for k in elites_idxs]
        elites_weights = torch.cat(
            [w.clone().detach().unsqueeze(0) for w in elites_weights], dim=0
        )
        centroid = elites_weights.mean(0)

        # Update covariance
        matrix.update_noise()
        matrix.update_covariance(elites_weights)
        if cfg.verbose:
            print("---------------------")
    return best_score


# %%
@hydra.main(
    config_path="./configs/",
    config_name="cem_mountain_car.yaml",
    # config_name="cem_cartpole.yaml",
    # version_base="1.3",
)
def main(cfg_raw: DictConfig):
    torch.random.manual_seed(seed=cfg_raw.algorithm.seed.torch)

    if "optuna" in cfg_raw:
        launch_optuna(cfg_raw, run_cem)
    else:
        logger = Logger(cfg_raw)
        run_cem(cfg_raw, logger)


if __name__ == "__main__":
    main()
