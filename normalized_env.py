# nomalized_env.py
import gym
import numpy as np


class NormalizedEnv(gym.ActionWrapper)
    def __init__(self, env):
        super().__init__(env)
        
    def _action(self, action):
        # modify the action
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b
    
    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)