#!/usr/local/bin/python3
import gym
import torch
import numpy as np

from copy import deepcopy

#our implementations
from agent import DDPGAgent
from evaluator import Evaluator
from normalized_env import NormalizedEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(env, agent, evaluator, num_iterations, validate_steps, output, debug=False):
    max_episode_length = 1000 #max time per episode so the agent doesn't stall
    agent.is_training = True
    step = episode = episode_steps = 0 
    episode_reward = 0.0 # episode is each instance of the game running
    observation = None # current state

    warmup_steps = 100 # first 100 steps we do a random observation
    validate_steps = 2000 # every 2000 steps we evaluate the agent

    while step < num_iterations:
        if observation is None:
            observation, _ = deepcopy(env.reset())
            agent.reset(observation)
        
        # action selection
        if step <= warmup_steps:
            action = agent.random_action()
        else:
            action = agent.select_action(observation) 

        observation2, reward, terminated, truncated, info = env.step(action)
        observation2 = deepcopy(observation2)        

        if episode_steps >= (max_episode_length):
            terminated = True
        
        agent.observe(reward, observation2, (terminated or truncated))
        
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
            
            action = torch.tensor(agent.select_action(observation), dtype = torch.float32, device=device).unsqueeze(0)
            observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            next_state_mask = torch.zeros_like(observation)
            r_t = torch.tensor([0.0], dtype=torch.float32, device=device)
            terminated = torch.tensor([0.0], dtype=torch.float32, device=device)
            agent.memory.append(observation, action, r_t, next_state_mask, terminated)
            
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
    #agent.load_weights("saved_models")
    
    train(env=env, agent=agent, evaluator=None, num_iterations=400000, validate_steps=2000, output="saved_models/", debug=True)

