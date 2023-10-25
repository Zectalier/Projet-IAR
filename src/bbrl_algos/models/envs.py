import os
import gym

# import bbrl_gymnasium is necessary to see the bbrl_gymnasium environments
import bbrl_gymnasium

# import gym_torcs

from bbrl import get_arguments, get_class
from typing import Tuple
from bbrl.agents.gymnasium import make_env, GymAgent, ParallelGymAgent
from functools import partial

from bbrl_algos.wrappers.env_wrappers import MazeMDPContinuousWrapper


assets_path = os.getcwd() + "/../../assets/"


def get_eval_env_agent(cfg):
    eval_env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name, autoreset=False),
        cfg.algorithm.n_envs_eval,
        include_last_state=True,
        seed=cfg.algorithm.seed.eval,
    )
    return eval_env_agent


def get_eval_env_agent_rich(cfg):
    eval_env_agent = ParallelGymAgent(
        make_env_fn=get_class(cfg.gym_env_eval),
        num_envs=cfg.algorithm.n_envs_eval,
        make_env_args=get_arguments(cfg.gym_env_eval),
        seed=cfg.algorithm.seed.eval,
    )
    return eval_env_agent


def get_env_agents(
    cfg, *, autoreset=True, include_last_state=True
) -> Tuple[GymAgent, GymAgent]:
    # Returns a pair of environments (train / evaluation) based on a configuration `cfg`

    if "xml_file" in cfg.gym_env:
        xml_file = assets_path + cfg.gym_env.xml_file
        print("loading:", xml_file)
    else:
        xml_file = None

    if "wrappers" in cfg.gym_env:
        print("using wrappers:", cfg.gym_env.wrappers)
        # wrappers_name_list = cfg.gym_env.wrappers.split(',')
        wrappers_list = []
        wr = get_class(cfg.gym_env.wrappers)
        # for i in range(len(wrappers_name_list)):
        wrappers_list.append(wr)
        wrappers = wrappers_list
        print(wrappers)
    else:
        wrappers = []

    # Train environment
    if xml_file is None:
        train_env_agent = ParallelGymAgent(
            partial(
                make_env, cfg.gym_env.env_name, autoreset=autoreset, wrappers=wrappers
            ),
            cfg.algorithm.n_envs,
            include_last_state=include_last_state,
            seed=cfg.algorithm.seed.train,
        )

        # Test environment (implictly, autoreset=False, which is always the case for evaluation environments)
        eval_env_agent = ParallelGymAgent(
            partial(make_env, cfg.gym_env.env_name, wrappers=wrappers),
            cfg.algorithm.nb_evals,
            include_last_state=include_last_state,
            seed=cfg.algorithm.seed.eval,
        )
    else:
        train_env_agent = ParallelGymAgent(
            partial(
                make_env, cfg.gym_env.env_name, autoreset=autoreset, wrappers=wrappers
            ),
            cfg.algorithm.n_envs,
            include_last_state=include_last_state,
            seed=cfg.algorithm.seed.train,
        )

        # Test environment (implictly, autoreset=False, which is always the case for evaluation environments)
        eval_env_agent = ParallelGymAgent(
            partial(make_env, cfg.gym_env.env_name, wrappers=wrappers),
            cfg.algorithm.nb_evals,
            include_last_state=include_last_state,
            seed=cfg.algorithm.seed.eval,
        )

    return train_env_agent, eval_env_agent
