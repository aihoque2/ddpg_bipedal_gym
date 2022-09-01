#!/usr/local/bin/python3
import gym
import torch
import numpy as np

env = gym.make('BipedalWalker-v3')
print("here's observation space: ", env.observation_space)
print("here's action space: ", env.action_space)
env.reset()
while True:
    env.render()