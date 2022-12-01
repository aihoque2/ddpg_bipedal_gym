#!/usr/local/bin/python3
import gym
import numpy as np

num_steps = 1500

env = gym.make('BipedalWalker-v3', render_mode="human")
print("here's observation space: ", env.observation_space.shape[0])
print("here's action space: ", env.action_space.high[0])
obs = env.reset()

for step in range(num_steps):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        obs, info = env.reset()

env.close()