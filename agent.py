#!/usr/local/bin/python3

#agent.py
import gym
import time

import torch
import torch.nn as nn
from torch.optim import Adam

#local files
from model import Actor, Critic
from utils.replay_buffer import ReplayBuffer
from utils.update_funcs import *
from utils.noise_model import * 

class DDPG_Agent:
    def __init__(self, env, state_size, action_size, action_lim, hidden1, hidden2, prate, rate, device):

        self.state_size = state_size
        self.action_size = action_size
        self.action_lim = action_lim
        self.prate = prate # actor learning rate
        self.rate = rate # critic learn rate for optimizerr
        self.tau = 0.001 # tarrget update weight
        self.discount = 0.99

        # neural network setup
        self.actor = Actor(self.state_size, self.action_size, action_lim, hidden1, hidden2)
        self.actor_tgt = Actor(self.state_size, self.action_size, action_lim)
        self.actor_optim = Adam(self.actor.parameters, lr=self.prate)
        
        self.critic = Critic(self.state_size, self.action_size)
        self.critic_tgt = Critic(self.state_size, self.action_size)
        self.critic_optim  = Adam(self.critic.parameters(), lr=self.rate)

        hard_update(self.actor, self.actor_tgt)
        hard_update(self.critic, self.critic_tgt)

        self.memory = ReplayBuffer(7e3)    
        self.noise = OrnsteinUhlenbeckProcess(theta=0.15, sigma=0.2, mu=0.0, size=self.action_size, )

        self.depsilon = 1.0/50000.0

        self.epsilon = 1.0

        self.s_t = None
        self.a_t = None
        self.is_training = True

        if (torch.cuda.is_available()){
            self.cuda()
        }

    def update_policy(self):
        pass

    def eval(self):
        pass

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        pass

    def select_action(self, s_t, decay_epsilon=True):
        pass

    def reset(self):
        pass

    def load_weights(self, input):
        pass

    def save_model(self, output):
        pass

    def seed(self, s):
        torch.manual_seed(s)
        if (torch.cuda.is_available()):
            torch.cuda.manual_seed(s)


    def cuda(self):
        pass
        
