#!/usr/local/bin/python3

"""
replay_buffer.py

this file is to implement experience 
replay in DDPG, so our neural net can
take samples that are i.i.d.
"""

import torch
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state')) #data that the replaybuffer takes


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, s1, a1 , r1, s2):
        self.memory.append(Transition(s1, a1, r1, s2))

    def sample(self, batch_size)    :
        #return s1, a1, r1, s2
        batch = random.sample(self.memory, batch_size)
        
        s1 = [tuple.state for tuple in batch]
        a1 = [tuple.action for tuple in batch]
        r1 = [tuple.reward for tuple in batch]
        s2 = [tuple.next_state for tuple in batch]

        return torch.Tensor(s1), torch.Tensor(a1), torch.Tensor(r1), torch.Tensor(s2)
    
    def __len__(self):
        return len(self.memory)