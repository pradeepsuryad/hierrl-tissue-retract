"""Uniform replay buffer shared by every algorithm."""
import random
import numpy as np
import torch

from ..common import device

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=300_000):
        self.capacity = capacity
        self.ptr = self.size = 0
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),       dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),       dtype=np.float32)
    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (torch.FloatTensor(self.obs[idx]).to(device),
                torch.FloatTensor(self.actions[idx]).to(device),
                torch.FloatTensor(self.rewards[idx]).to(device),
                torch.FloatTensor(self.next_obs[idx]).to(device),
                torch.FloatTensor(self.dones[idx]).to(device))
    def __len__(self): return self.size
