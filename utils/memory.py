#!/usr/local/bin/python3

"""
memory.py

this file is to implement experience 
replay in DDPG, so our neural net can
take samples that are i.i.d.
"""

import torch
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done')) #data that the replaybuffer takes

class RingBuffer:
    def __init__(self, capacity):
        #TODO 
        self.maxlen = capacity
        self.length = 0
        pass

    def __len__(self):
        #TODO 
        pass
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError
        pass


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, s1, a1 , r1, s2, done):
        self.memory.append(Transition(s1, a1, r1, s2, done))

    def sample(self, batch_size)    :
        #return s1, a1, r1, s2
        batch = random.sample(self.memory, batch_size)
        
        s1 = [tuple.state for tuple in batch]
        a1 = [tuple.action for tuple in batch]
        r1 = [tuple.reward for tuple in batch]
        s2 = [tuple.next_state for tuple in batch]
        done = [0.0 if tuple.done else 1.0 for tuple in batch] 

        return torch.Tensor(s1), torch.Tensor(a1), torch.Tensor(r1), torch.Tensor(s2), torch.Tensor(done)
    
    def __len__(self):
        return len(self.memory)


class Memory:
    def __init__(self):
        pass