import hydra
import torch
from omegaconf import DictConfig

from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from bbrl.agents.gymnasium import make_env, ParallelGymAgent
from functools import partial

from bbrl_algos.models.tux_actors import ContinuousActor


@hydra.main(
    config_path="./configs/",
    config_name="cem_tux.yaml",
)
def main(cfg: DictConfig):

    make_stkenv = make_env(
        "supertuxkart-v0", render_mode="human", num_kart=5, use_ai=False, autoreset=True
    )
    # make_stkenv = make_env("supertuxkart-flattened-continuous-actions-v0",
    #                      render_mode="human", num_kart=5, use_ai=False, autoreset=False, track="sandtrack")
    env_agent = ParallelGymAgent(make_stkenv, 1)
    filename = "supertuxkart-flattened-continuous-actions-v0cem2396.781005859375.agt"
    pilot = torch.load(filename)
    visu_agent = TemporalAgent(Agents(env_agent, pilot))
    workspace = Workspace()
    for epoch in range(10_000):

        if epoch == 0:
            visu_agent(workspace, n_steps=100, render=True)
        else:
            workspace.copy_n_last_steps(1)
            visu_agent(workspace, t=1, n_steps=100, render=True)
    # visu_agent(workspace, t=0, stop_variable="env/done", render=True)
