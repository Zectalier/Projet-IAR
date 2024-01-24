import gymnasium as gym

# For permission issues on the machine.
import sys
sys.path.append('/users/nfs/Etu7/21201287/Documents/Projet-IAR/')

from stable_baselines3.droQ.droQ import DroQ
from stable_baselines3.common.logger import configure

from gymnasium.wrappers.monitoring import video_recorder


model = DroQ.load("droq_hopper")

env = gym.make("Hopper-v4", render_mode='human')

env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
# Save the video

env.render('human')

env.close()
