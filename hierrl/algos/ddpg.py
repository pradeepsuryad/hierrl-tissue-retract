"""DDPG with Ornstein-Uhlenbeck exploration (Lillicrap et al. 2016)."""
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..common import device
from .networks import DeterministicActor, DoubleQCritic
from .replay import ReplayBuffer

class OUNoise:
    def __init__(self, act_dim, sigma=0.2, theta=0.15, dt=1e-2):
        self.act_dim = act_dim; self.sigma = sigma; self.theta = theta; self.dt = dt; self.reset()
    def sample(self):
        self.x += -self.theta*self.x*self.dt + self.sigma*np.sqrt(self.dt)*np.random.randn(self.act_dim)
        return self.x.copy()
    def reset(self): self.x = np.zeros(self.act_dim)

class DDPG:
    def __init__(self, obs_dim, act_dim, lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.99, tau=0.005, batch_size=128, warmup_steps=1000, buffer_capacity=300_000):
        self.gamma = gamma; self.tau = tau
        self.batch_size = batch_size; self.warmup_steps = warmup_steps
        self.actor         = DeterministicActor(obs_dim, act_dim).to(device)
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic        = DoubleQCritic(obs_dim, act_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.noise  = OUNoise(act_dim)
        self.buffer = ReplayBuffer(obs_dim, act_dim, buffer_capacity)
    def select_action(self, obs, deterministic=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad(): a = self.actor(obs_t).cpu().numpy()[0]
        return a if deterministic else np.clip(a + self.noise.sample(), -1.0, 1.0)
    def update(self):
        if len(self.buffer) < self.warmup_steps: return {}
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        with torch.no_grad():
            na = self.actor_target(next_obs)
            q_target = rewards + self.gamma*(1-dones)*torch.min(*self.critic_target(next_obs, na))
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad(); critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0); self.critic_opt.step()
        actor_loss = -self.critic.q_min(obs, self.actor(obs)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.actor_opt.step()
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}
