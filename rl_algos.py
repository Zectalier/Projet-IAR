import gymnasium as gym

import sys
sys.path.append('/users/nfs/Etu7/21201287/Documents/Projet-IAR/')

from stable_baselines3.droQ.droQ import DroQ
from stable_baselines3.droQ.redq import REDQ
from stable_baselines3.sac import SAC


from wandb.integration.sb3 import WandbCallback
import wandb

from gymnasium.wrappers.monitoring import video_recorder

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure

import argparse

def main(args):
    config = {
        'env': args.environment,
        'total_timesteps': args.steps,
        'learning_rate': float(args.learning_rate),
        'gradient_steps': 20,
        'policy': 'MlpPolicy',
        'net_arch': [400, 300],
        'dropout_rate': float(args.dropout),
        'wandb': True,
        'verbose': 1
    }

    run = wandb.init(project='Project-IAR', entity='osmane', name=f'{args.algorithm}-{args.environment}', config=config, tags=[args.algorithm, args.environment, 'MlpPolicy', '256x256'],
        reinit=True,
        monitor_gym=True,
        sync_tensorboard=True,
        save_code=True)
    
    pk = None
    
    if args.algorithm != 'SAC':
        pk = {
            'net_arch': config['net_arch'],
            'dropout_rate': config['dropout_rate'],
            'n_critics': int(args.n_critics),
        }
    else:
        pk = {
            'net_arch': config['net_arch'],
            'n_critics': int(args.n_critics)
        }

    new_logger = configure(f'./logs/{args.algorithm}_{args.environment}_{run.id}.data', ["stdout", "csv", "tensorboard"])

    env = gym.make(config['env'])

    #env = gym.wrappers.RecordVideo(env, 'video_walker', episode_trigger = lambda x: x % 5 == 0)
    
    if args.algorithm == 'SAC':
        model = SAC('MlpPolicy', 
                    env, 
                    verbose= config['verbose'],
                    learning_rate= config['learning_rate'],
                    gradient_steps= config['gradient_steps'],
                    policy_kwargs= pk, 
                    tensorboard_log=f"runs/{run.id}",
                    target_entropy= -3.0)
    elif args.algorithm == 'DroQ':
        model = DroQ('MlpPolicy', 
                    env, 
                    verbose= config['verbose'],
                    learning_rate= config['learning_rate'],
                    gradient_steps= config['gradient_steps'],
                    policy_kwargs= pk, 
                    tensorboard_log=f"runs/{run.id}",
                    target_entropy= -3.0)
    else:
        model = REDQ('MlpPolicy', 
                    env, 
                    verbose= config['verbose'],
                    learning_rate= config['learning_rate'],
                    gradient_steps= config['gradient_steps'],
                    policy_kwargs= pk, 
                    tensorboard_log=f"runs/{run.id}",
                    target_entropy= -3.0)

    model.set_logger(new_logger)

    model.learn(config['total_timesteps'],
                callback=WandbCallback(verbose=2,
                                       gradient_save_freq=100000), 
                log_interval=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", choices=['DroQ', 'SAC', 'REDQ'], required=True, help="Choose the algorithm to use")
    parser.add_argument("-s", "--steps", type=int, default=100_000, help="Learning steps to perform")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.03, help="Learning rate")
    parser.add_argument("-d", "--dropout", type=float, default=0.01, help="Dropout rate")
    parser.add_argument("-env", "--environment", default="Walker2d-v4", help="Gym environment to use")
    parser.add_argument("-c", "--n_critics", default="2", help="The number of critics to use")
    parser.add_argument("-g", "--gradient_steps", default="20", help="The number of gradient steps to perform")

    args = parser.parse_args()
    main(args)