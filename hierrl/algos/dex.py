"""DEX: SAC plus an annealed behaviour-cloning loss on scripted demos."""
import random
import numpy as np
import torch
import torch.nn.functional as F

from ..common import device
from .sac import SAC

class DEX:
    def __init__(self, obs_dim, act_dim,
                 lambda_bc_start=1.0, lambda_bc_end=0.0, anneal_steps=30000, **sac_kwargs):
        self.sac = SAC(obs_dim, act_dim, **sac_kwargs)
        self.lambda_start = lambda_bc_start; self.lambda_end = lambda_bc_end
        self.anneal_steps = anneal_steps; self._step = 0
        self.demo_indices = []
    @property
    def buffer(self): return self.sac.buffer
    def add_demo(self, obs, action, reward, next_obs, done):
        idx = self.sac.buffer.ptr
        self.sac.buffer.add(obs, action, reward, next_obs, done)
        self.demo_indices.append(idx)
    def select_action(self, obs, deterministic=False):
        return self.sac.select_action(obs, deterministic)
    def _lambda(self):
        t = min(self._step / max(self.anneal_steps, 1), 1.0)
        return self.lambda_start + t*(self.lambda_end - self.lambda_start)
    def update(self):
        buf = self.sac.buffer
        if len(buf) < self.sac.warmup_steps: return {}
        self._step += 1; lam = self._lambda()
        obs, actions, rewards, next_obs, dones = buf.sample(self.sac.batch_size)
        with torch.no_grad():
            na, nlp, _ = self.sac.actor.sample(next_obs)
            q_next   = torch.min(*self.sac.critic_target(next_obs, na)) - self.sac.alpha * nlp
            q_target = rewards + self.sac.gamma*(1-dones)*q_next
        q1, q2 = self.sac.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.sac.critic_opt.zero_grad(); critic_loss.backward()
        nn.utils.clip_grad_norm_(self.sac.critic.parameters(), 1.0); self.sac.critic_opt.step()
        new_a, log_pi, _ = self.sac.actor.sample(obs)
        sac_loss = (self.sac.alpha*log_pi - self.sac.critic.q_min(obs, new_a)).mean()
        bc_loss  = torch.tensor(0.0, device=device)
        if lam > 0 and len(self.demo_indices) > 0:
            d_idx  = np.random.choice(self.demo_indices, min(64, len(self.demo_indices)))
            d_obs  = torch.FloatTensor(buf.obs[d_idx]).to(device)
            d_acts = torch.FloatTensor(buf.actions[d_idx]).to(device)
            pred, _, _ = self.sac.actor.sample(d_obs)
            bc_loss = F.mse_loss(pred, d_acts)
        actor_loss = sac_loss + lam*bc_loss
        self.sac.actor_opt.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.sac.actor.parameters(), 1.0); self.sac.actor_opt.step()
        aloss = -(self.sac.log_alpha*(log_pi + self.sac.target_entropy).detach()).mean()
        self.sac.alpha_opt.zero_grad(); aloss.backward(); self.sac.alpha_opt.step()
        self.sac.alpha = self.sac.log_alpha.exp().item()
        for p, tp in zip(self.sac.critic.parameters(), self.sac.critic_target.parameters()):
            tp.data.copy_(self.sac.tau*p.data + (1-self.sac.tau)*tp.data)
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item(),
                'bc_loss': bc_loss.item(), 'lambda_bc': lam}
