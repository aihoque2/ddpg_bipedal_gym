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
from utils.helper_funcs import *
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
        #TODO
        #update the actor. VERY IMPORTANT!

        pass

    def eval(self):
        self.actor.eval()
        self.actor_tgt.eval()

        self.critic.eval()
        self.critic_tgt.eval()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        """
        action taken in exploration phase
        """
        action = np.random(-1., 1., self.action_size)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(self.action(to_tensor(np.array([s_t])))).squeeze(0)

        #add the noise component to this action
        action += self.is_training*max(0, self.epsilon)*self.noise.sample() 
        
        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, input):
        if input is None: return
        #load actor and critic weights both
        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(input))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(input))
        )

    def save_model(self, output):
        #save both actor and critic models
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self, s):
        torch.manual_seed(s)
        if (torch.cuda.is_available()):
            torch.cuda.manual_seed(s)


    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()
