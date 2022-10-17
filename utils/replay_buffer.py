#!/usr/local/bin/python3

"""
replay_buffer.py

this file is to implement experience 
replay in DDPG, so our neural net can
take samples that are i.i.d.
"""

import torch
import gym
import random
from collections import namedtuple, deque

Transition = namedtupled('Transition', ('state', 'action', 'next_state', 'reward')) #data that the replaybuffer takes


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)