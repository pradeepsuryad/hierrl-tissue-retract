"""Soft Actor-Critic (Haarnoja et al. 2018)."""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..common import device
from .networks import GaussianActor, DoubleQCritic
from .replay import ReplayBuffer

class SAC:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, tau=0.005,
                 batch_size=256, warmup_steps=1000, buffer_capacity=300_000):
        self.gamma = gamma; self.tau = tau
        self.batch_size = batch_size; self.warmup_steps = warmup_steps
        self.act_dim = act_dim
        self.actor         = GaussianActor(obs_dim, act_dim).to(device)
        self.critic        = DoubleQCritic(obs_dim, act_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.target_entropy = -act_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha     = self.log_alpha.exp().item()
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt  = optim.Adam([self.log_alpha],          lr=lr)
        self.buffer = ReplayBuffer(obs_dim, act_dim, buffer_capacity)
    def select_action(self, obs, deterministic=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            a, _, mu = self.actor.sample(obs_t)
        return (mu if deterministic else a).cpu().numpy()[0]
    def update(self):
        if len(self.buffer) < self.warmup_steps: return {}
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        with torch.no_grad():
            na, nlp, _ = self.actor.sample(next_obs)
            q_next = torch.min(*self.critic_target(next_obs, na)) - self.alpha * nlp
            q_target = rewards + self.gamma * (1 - dones) * q_next
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad(); critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0); self.critic_opt.step()
        new_a, log_pi, _ = self.actor.sample(obs)
        actor_loss = (self.alpha * log_pi - self.critic.q_min(obs, new_a)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.actor_opt.step()
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1-self.tau) * tp.data)
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item(), 'alpha': self.alpha}
