#!/usr/local/bin/python3
import gym
import numpy as np
from model import Actor
import torch
import torch.nn as nn

num_steps = 1500

env = gym.make('BipedalWalker-v3', render_mode="human")
print("here's observation space: ", env.observation_space.shape)
print("here's action space: ", type(env.action_space.high[0]))

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
obs, info = env.reset()
hidden1 = 400
hidden2 = 300
fc1 = nn.Linear(state_size, hidden1) #input layer that takes in our read state		
fc2 = nn.Linear(hidden1, hidden2)
#action = actor(torch.from_numpy(obs))

for step in range(num_steps):
    action = env.action_space.sample()

    if step>10:
        print("action from linear layer")
        action = fc1(torch.from_numpy(obs))

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        obs, info = env.reset()

env.close()