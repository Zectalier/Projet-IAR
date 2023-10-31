import hydra
import optuna
import yaml

from omegaconf import DictConfig
from bbrl import get_arguments, get_class
from bbrl_algos.models.loggers import Logger


# %%
def get_trial_value(trial: optuna.Trial, cfg: DictConfig, variable_name: str):
    suggest_type = cfg["suggest_type"]
    args = cfg.keys() - ["suggest_type"]
    args_str = ", ".join([f"{arg}={cfg[arg]}" for arg in args])
    return eval(f'trial.suggest_{suggest_type}("{variable_name}", {args_str})')


def get_trial_config(trial: optuna.Trial, cfg: DictConfig):
    for variable_name in cfg.keys():
        if type(cfg[variable_name]) != DictConfig:
            continue
        else:
            if "suggest_type" in cfg[variable_name].keys():
                cfg[variable_name] = get_trial_value(
                    trial, cfg[variable_name], variable_name
                )
            else:
                cfg[variable_name] = get_trial_config(trial, cfg[variable_name])
    return cfg


def launch_optuna(cfg_raw, run_func):
    cfg_optuna = cfg_raw.optuna

    def objective(trial):
        cfg_sampled = get_trial_config(trial, cfg_raw.copy())

        logger = Logger(cfg_sampled)
        try:
            trial_result: float = run_func(cfg_sampled, logger, trial)
            logger.close()
            return trial_result
        except optuna.exceptions.TrialPruned:
            logger.close()
            return float("-inf")

    study = hydra.utils.call(cfg_optuna.study)
    study.optimize(func=objective, **cfg_optuna.optimize)

    file = open("best_params.yaml", "w")
    yaml.dump(study.best_params, file)
    file.close()
