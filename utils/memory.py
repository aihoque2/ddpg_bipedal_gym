import torch
from collections import namedtuple, deque
import random



Experience = namedtuple("Experience", 'state, action, reward, next_state, done')

class ReplayMemory():
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size

    def __len__(self):
        return len(self.buffer)
    
    def append(self, state, action, reward, next_state, done):
        assert(torch.is_tensor(state), f"param 'state' is not of type 'torch.Tensor' \n state's type: {type(state)}")
        assert(torch.is_tensor(action), f"param 'action' is not of type 'torch.Tensor' \n action's type: {type(action)}")
        assert(torch.is_tensor(reward), f"param 'reward' is not of type 'torch.Tensor' \n reward's type: {type(reward)}")
        assert(torch.is_tensor(next_state), f"param 'action' is not of type 'torch.Tensor' \n action's type: {type(next_state)}")


        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        count = min(len(self.buffer), batch_size)
        experiences = random.sample(self.buffer, batch_size)
        batch = Experience(*zip(*experiences))

        state_batch = torch.cat(batch.state)
        print("here's state_batch: ", type(state_batch))
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        terminal_batch = torch.cat(batch.done)

        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch                
