import gymnasium as gym

from stable_baselines3.sac import SAC
from wandb.integration.sb3 import WandbCallback
import wandb

from gymnasium.wrappers.monitoring import video_recorder

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure

config = {
    'env': 'Hopper-v4',
    'total_timesteps': 400_000,
    'learning_rate': 3e-3,
    'gradient_steps': 20,
    'policy': 'MlpPolicy',
    'net_arch': [400, 300],
    'dropout_rate': 0.001,
    'wandb': True,
    'verbose': 1
}

for i in range(15):
    # Init wandb project 'Project-IAR' for 'SAC-Hopper-v4' experiment.
    run = wandb.init(project='Project-IAR', entity='osmane', name='SAC-Hopper-v4', config=config, tags=['SAC', 'Hopper-v4', 'MlpPolicy', '256x256', 'dropout_rate=0.01'],
        reinit=True,
        monitor_gym=True,
        sync_tensorboard=True,
        save_code=True)


    pk = {
        'net_arch': config['net_arch'],
    }
    
    new_logger = configure(f'./logs/SAC_hopper-v4_{run.id}.data', ["stdout", "csv", "tensorboard"])
    
    env = gym.make(config['env'])

    #env = gym.wrappers.RecordVideo(env, 'video_walker', episode_trigger = lambda x: x % 5 == 0)

    model = SAC('MlpPolicy', 
                env, 
                verbose= config['verbose'],
                learning_rate= config['learning_rate'],
                gradient_steps= config['gradient_steps'],
                policy_kwargs= pk, 
                tensorboard_log=f"runs/{run.id}",
                target_entropy= -1.0)
    
    model.set_logger(new_logger)

    model.learn(config['total_timesteps'],
                callback=WandbCallback(verbose=2,
                                       gradient_save_freq=100,
                                       model_save_path=f"models/{run.id}"), 
                log_interval=1)