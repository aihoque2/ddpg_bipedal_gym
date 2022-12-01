"""
evaluator.py
this file is created to calculate the reward
recived after each simulation run when training
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

class Evaluator:
    def __init__(self, num_episodes, interval, save_path='', max_episode_length=None):
        self.num_episodes = num_episodes
        self.interval = interval # number of validation steps
        self.max_episode_length= max_episode_length
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes,0)

    def __call__(self, env, policy, debug=False, visualize=False, save=True):
        """
        override the () operator
        
        @param policy: the function pointer to the agent's policy to call when testing our policy
        """
        self.is_training = False
        observation = None
        result = []

        for episode in range(self.num_episodes):

            #reset env tp start
            observation, info = env.reset()
            print("here's observation after env.reset(): ", observation.shape)
            episode_step = 0
            episode_reward = 0.0

            assert observation is not None

            done = False
            while not done:

                action = policy(observation) # use our policy to get an action

                observation, reward, terminated, truncated, info = env.step(action)

                if self.max_episode_length and episode_step >= (self.max_episode_length - 1):
                    done = True

                if visualize:
                    env.render(mode="human")

                #updates
                episode_reward += reward
                episode_step += 1

            # debugging
            statement = '[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward) 
            print("\033[93m {}\033[00m" .format(statement)) # print the episode/episode reward in yellow

            result.append(episode_reward)
        
        result = np.array(result).reshape(-1, 1)
        self.results = np.hstack([self.results, result]) #add the new result of the episode

        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))

        result.append(episode_reward)
        return np.mean(result)

    def save_results(self, name):
        """
        create a plot and save it
        """
        y = np.mean(self.results, axis=0)
        error=np.std(self.results, axis=0)
                    
        x = range(0,self.results.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(name + '.png')
        savemat(name + '.mat', {'reward':self.results})