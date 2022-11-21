#!/usr/local/bin/python3
import gym
import torch
import numpy as np

env = gym.make('BipedalWalker-v3', render_mode="human")

print("here's observation space: ", env.observation_space)
print("here's action space: ", env.action_space)
observation, info = env.reset(seed=42)
while True:
    env.render()