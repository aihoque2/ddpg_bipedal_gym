#!/usr/local/bin/python3
import gym
import torch
import numpy as np

from copy import deepcopy

def train(env, agent, evaluator, num_iterations, validate_steps, output, max_episode_length=None, debug=False):
    agent.is_training = True
    step = episode = episode_steps = 0 
    episode_reward = 0.0 # episode is each instance of the game running
    observation = None


    warmup_steps = 100 # first 100 steps we do a random observation
    validate_steps = 2000 # every 2000 steps we evaluate the agent


    while step < num_iterations:
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)
        
        if step <= warmup_steps:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)

        observation2, reward, terminated, truncated, info = env.step(action)
        observation2 = deepcopy(observation2)

        if episode_steps >= max_episode_length - 1:
            terminated = True

        agent.observe()

        if done:
            episode += 1
            episode_steps = 0
            episode_reward = 0.0
            observation = None



env = gym.make('BipedalWalker-v3', render_mode="human")

print("here's observation space: ", env.observation_space)
print("here's action space: ", env.action_space)
observation, info = env.reset(seed=42)
while True:
    env.render()