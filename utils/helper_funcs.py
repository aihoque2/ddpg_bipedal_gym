"""
helper_funcs.py

file used to hold helper functions
"""

import torch
import numpy as np

#update functions
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


# TODO: Remove
def to_numpy(var):
    return var.cpu().data.numpy() if not USE_CUDA else var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=np.float64):
    return torch.from_numpy(ndarray)