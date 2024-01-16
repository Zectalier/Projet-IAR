import gymnasium as gym

# For permission issues on the machine.
import sys
sys.path.append('/users/nfs/Etu7/21201287/Documents/Projet-IAR/')

from stable_baselines3.droQ.droQ import DroQ
from stable_baselines3.common.logger import configure
from gymnasium.wrappers.monitoring import video_recorder

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

pk = {
    'dropout_rate': 0.01,
    'net_arch': [32, 32],
}

env = gym.make("Walker2d-v4")

#env = gym.wrappers.RecordVideo(env, 'video_walker', episode_trigger = lambda x: x % 5 == 0)

model = DroQ('MlpPolicy', 
             env, 
             verbose= 1,
             learning_rate= 3e-3,
             gradient_steps= 20,
             policy_kwargs= pk)

model.learn(total_timesteps=1_000_000)
