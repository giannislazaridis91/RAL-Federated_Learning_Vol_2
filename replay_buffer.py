import numpy as np
import random
from collections import namedtuple, deque
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:

    # Initiliazie the ReplayBuffer.
    def __init__(self, Buffer_Size = 10000, Batch_Size = 64):
        self.mem_count = 0
        self.buffer_size = Buffer_Size
        self.batch_size = Batch_Size
        self.buffer_size_count = 0
        self.experiences = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "next_action", "done"])
        self.memory = deque(maxlen=self.buffer_size)
    
    # Store a transition.
    def store_transition(self, state, action, reward, _state, _action, done):
        e = self.experiences(state, action, reward, _state, _action, done)
        self.memory.append(e)
    
    # Collect a minibatch of Batch_Size number of transitions.
    def minibatch(self):
        experiences = random.sample(self.memory,k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        return(states)