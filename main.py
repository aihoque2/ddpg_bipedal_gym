#!/usr/local/bin/python3
import gym
import torch
import numpy as np

from copy import deepcopy

#our implementations
from agent import DDPGAgent
from evaluator import Evaluator
from normalized_env import NormalizedEnv

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

def train(env, agent, evaluator, num_iterations, validate_steps, output, debug=False):
    max_episode_length = 500
    agent.is_training = True
    step = episode = episode_steps = 0 
    episode_reward = 0.0 # episode is each instance of the game running
    observation = None # current state


    warmup_steps = 100 # first 100 steps we do a random observation
    validate_steps = 2000 # every 2000 steps we evaluate the agent


    while step < num_iterations:
        print("here's step: ", step)
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)
        
        # action selection
        if step <= warmup_steps:
            action = agent.random_action()
        else:
            action = agent.select_action(observation) 

        observation2, reward, terminated, truncated, info = env.step(action)
        observation2 = deepcopy(observation2)        

        if episode_steps >= (max_episode_length - 1):
            terminated = True
        
        print("here's is_training before observe() call: ", agent.is_training)
        agent.observe(reward, observation2, terminated)
        
        if step > warmup_steps:
            agent.optimize()


        # evaluation
        if evaluator is not None and step != 0 and step % validate_steps == 0:
            print("in evaluator")
            agent.is_training=False
            policy = lambda x : agent.select_action(x, decay_epsilon = False)
            validate_reward = evaluator(env, policy, debug=True, visualize=False, save=True)
            statement = '[Evaluate] step {}: validate_reward:{}'.format(step, validate_reward) 
            print("\033[93m {}\033[00m" .format(statement))
            agent.is_training=True            

        #update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if terminated: # end of an episode
            if debug: 
                statement = '#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step)
                print("\033[92m {}\033[00m" .format(statement))
            
            agent.memory.append(observation, agent.select_action(observation), 0.0, terminated)
            
            episode += 1
            episode_steps = 0
            episode_reward = 0.0
            observation = None
        


    agent.save_model(output)

if __name__ == "__main__":    
    env = gym.make('BipedalWalker-v3', render_mode="human")
    #norm_env = NormalizedEnv(env)

    state_size = int(env.observation_space.shape[0])
    action_size = env.action_space.shape[0]
    action_lim = env.action_space.high[0]

    agent = DDPGAgent(env, state_size, action_size, action_lim, prate=0.0001, rate=0.001)
    evaluator = Evaluator(num_episodes=20, interval=2000, save_path="saved_models/output.pth")
    
    train(env=env, agent=agent, evaluator=evaluator, num_iterations=200000, validate_steps=2000, output="saved_models/output.pth", debug=True)

