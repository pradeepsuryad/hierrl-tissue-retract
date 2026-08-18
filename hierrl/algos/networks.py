"""MLP trunk, Gaussian/deterministic actors, twin-Q critic and value net."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256,256), activation=nn.ReLU, output_activation=None):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), activation()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        if output_activation: layers.append(output_activation())
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

LOG_STD_MIN, LOG_STD_MAX = -5, 2

class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.trunk       = MLP(obs_dim, hidden[-1], hidden[:-1])
        self.mu_head     = nn.Linear(hidden[-1], act_dim)
        self.logstd_head = nn.Linear(hidden[-1], act_dim)
    def forward(self, obs):
        h = self.trunk(obs)
        return self.mu_head(h), self.logstd_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
    def sample(self, obs):
        mu, logstd = self.forward(obs)
        dist = torch.distributions.Normal(mu, logstd.exp())
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = (dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mu)

class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.net = MLP(obs_dim, act_dim, hidden, output_activation=nn.Tanh)
    def forward(self, obs): return self.net(obs)

class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.q1 = MLP(obs_dim+act_dim, 1, hidden)
        self.q2 = MLP(obs_dim+act_dim, 1, hidden)
    def forward(self, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)
    def q_min(self, obs, action):
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)

class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=(256,256)):
        super().__init__()
        self.net = MLP(obs_dim, 1, hidden)
    def forward(self, obs): return self.net(obs)
