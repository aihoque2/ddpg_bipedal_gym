#!/usr/local/bin/python3

# agent.py
import gym
import time

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

# local files
from model import Actor, Critic
from utils.memory import ReplayMemory
from utils.helper_funcs import *
from utils.noise_model import * 

LEARNING_RATE = 0.001
GAMMA = 0.99 # discount rate
TAU = 0.001 # soft target update factor

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, env, state_size, action_size, action_lim, prate, rate):
        
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.action_lim = action_lim
        self.prate = prate # actor learning rate
        self.rate = rate # critic learn rate for optimizerr
        self.tau = 0.001 # tarrget update weight
        self.discount = GAMMA
        self.batch_size = 64

        # neural network setup
        self.actor = Actor(self.state_size, self.action_size, action_lim)
        self.actor_tgt = Actor(self.state_size, self.action_size, action_lim)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.prate)
        
        self.critic = Critic(self.state_size, self.action_size)
        self.critic_tgt = Critic(self.state_size, self.action_size) # used to calculate y_i
        self.critic_optim  = Adam(self.critic.parameters(), lr=self.rate)

        hard_update(self.actor, self.actor_tgt)
        hard_update(self.critic, self.critic_tgt)

        self.memory = ReplayMemory(size=int(7e6))   
        self.noise_model = OrnsteinUhlenbeckProcess(theta=0.15, sigma=0.2, mu=0.0, size=self.action_size, )

        self.depsilon = 1.0/50000.0

        self.epsilon = 1.0

        self.s_t = None # put this as a torch Tensor
        self.a_t = None # put this as a torch Tensor
        self.is_training = True

        if (torch.cuda.is_available()):
            self.cuda()

    def optimize(self):
        """TODO"""
        # get the new action and rreward for experrience replay
        s1, a1, r1, s2, terminal_batch = self.memory.sample(self.batch_size)
                
        # critic optimization
        a2 = self.actor_tgt.forward(s1)

        y_i = r1 + self.discount*terminal_batch*torch.squeeze(self.critic_tgt.forward(s2, a2)) # why we need crtic_tgt
        y_predicted = torch.squeeze(self.critic.forward(s1, a1))

        loss_critic = F.smooth_l1_loss(y_predicted, y_i) 
        self.critic_optim.zero_grad() #reset gradients in the optimizer
        loss_critic.backward() # backpropogate the loss 
        self.critic_optim.step()

        # actor optimization
        loss_actor = -1*torch.sum(self.critic.forward(s1, self.actor.forward(s1))) #we wanna max this value, so we trick the optimizer by calculating -1
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
        
        soft_update(self.critic_tgt, self.critic, TAU)
        soft_update(self.actor_tgt, self.actor, TAU)

    def eval(self):
        self.actor.eval()
        self.actor_tgt.eval()

        self.critic.eval()
        self.critic_tgt.eval()

    def observe(self, r_t, s_t2, terminated):
        """
        add the new experience into experience replay
        """
        if self.is_training:
            s_t2 = torch.tensor(s_t2, dtype=torch.float32, device=device).unsqueeze(0)
            r_t = torch.tensor([r_t], dtype=torch.float32, device=device)
            terminated = torch.tensor([0.0 if terminated else 1.0], dtype=torch.float32, device=device)
            self.memory.append(self.s_t, self.a_t, r_t, s_t2, terminated)
            self.s_t = s_t2
        
    def random_action(self):
        """     
        action taken in exploration phase
        """
        #action = np.random.uniform(-1., 1., self.action_size)
        action = self.env.action_space.sample()
        self.a_t = torch.tensor(action, dtype = torch.float32, device=device).unsqueeze(0)
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = self.actor(torch.from_numpy(s_t))

        action = action.detach().numpy()

        #add the noise component to this action
        action += self.is_training*max(0, self.epsilon)*self.noise_model.sample() 
        
        if decay_epsilon:
            self.epsilon -= self.depsilon

        self.a_t = action
        return action

    def reset(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        self.s_t = obs
        self.noise_model.reset_states()

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
        self.actor_tgt.cuda()
        self.critic.cuda()
        self.critic_tgt.cuda()
