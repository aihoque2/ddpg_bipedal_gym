#!/usr/local/bin/python3

#noise_model.py
from abc import ABC, abstractclassmethod
import numpy as np

class RandomProcess(ABC):
    @abstractclassmethod
    def reset_states():
        pass


class OrnsteinUhlenbeckProcess(RandomProcess):
    """
    TODO:
    Noise process used for the policy 
    when generating an action when learnring
    """
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        self.mu = mu
        self.sigma=sigma
        self.n_steps = n_steps_annealing