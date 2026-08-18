"""ViSkill: four SAC sub-policies chained by a shared value network."""
from collections import deque

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from ..common import device
from .networks import ValueNet
from .sac import SAC

class ViSkill:
    def __init__(self, obs_dim, act_dim, n_skills=4, lr=3e-4, gamma=0.99, tau=0.005,
                 batch_size=128, warmup_steps=500, buffer_capacity=150_000):
        self.n_skills = n_skills; self.obs_dim = obs_dim; self.act_dim = act_dim
        self.skills = [SAC(obs_dim, act_dim, lr=lr, gamma=gamma, tau=tau,
                           batch_size=batch_size, warmup_steps=warmup_steps,
                           buffer_capacity=buffer_capacity)
                       for _ in range(n_skills)]
        self.value_net = ValueNet(obs_dim).to(device)
        self.value_opt = optim.Adam(self.value_net.parameters(), lr=lr)
        self.v_thresholds = [-2.0, -1.0, 0.0]
        self._v_history   = deque(maxlen=2000)
        self._meta_reliable = False
        self._meta_steps    = 0
        self.current_skill  = 0

    @staticmethod
    def _skill_reward(info, phase):
        d  = info.get('dist_grasp', 1.0)
        td = info.get('tissue_disp', 0.0)
        g  = info.get('grasped', False)
        f  = info.get('force', 0.0)
        fp = -0.3 * float(np.log(1 + max(0.0, f - 5.0)))
        if phase == 0: return float(-d * 0.5 + fp)
        elif phase == 1: return float(5.0 if g else -d * 0.3 + fp)
        elif phase == 2: return float(td / 0.15 * 0.5 + fp)
        else: return float((1.0 if td > 0.09 else -0.2) + fp)  # BUG FIX 1: 0.12 -> 0.09

    def select_action(self, obs, env_phase=None, deterministic=False):
        if deterministic and self._meta_reliable:
            skill = self._meta_select(obs)
        elif env_phase is not None:
            skill = min(int(env_phase), self.n_skills - 1)
        else:
            skill = self.current_skill
        self.current_skill = skill
        action = self.skills[skill].select_action(obs, deterministic=deterministic)
        # BUG FIX 3: when GRASP sub-agent is active, bias jaw strongly toward closing
        if skill == 1 and not deterministic:
            action[3] = np.clip(action[3] - 0.5, -1.0, 1.0)
        return action

    def _meta_select(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad(): v = self.value_net(obs_t).item()
        self._v_history.append(v)
        skill = sum(1 for thresh in self.v_thresholds if v > thresh)
        return min(skill, self.n_skills - 1)

    def _update_thresholds(self):
        if len(self._v_history) > 200:
            v = np.array(self._v_history)
            self.v_thresholds = [float(np.percentile(v, p)) for p in [25, 50, 75]]

    def store(self, obs, action, reward, next_obs, done, env_phase, info):
        skill_id   = min(int(env_phase), self.n_skills - 1)
        sub_reward = self._skill_reward(info, skill_id)
        self.skills[skill_id].buffer.add(obs, action, sub_reward, next_obs, float(done))
        self.skills[0].buffer.add(obs, action, reward, next_obs, float(done))

    def update(self, env_phase=0):
        skill_id = min(int(env_phase), self.n_skills - 1)
        logs = self.skills[skill_id].update()
        buf = self.skills[0].buffer
        if len(buf) >= 256:
            obs_b, _, rews, next_obs_b, dones_b = buf.sample(256)
            with torch.no_grad():
                v_target = rews + 0.99*(1-dones_b)*self.value_net(next_obs_b)
            v_loss = F.mse_loss(self.value_net(obs_b), v_target)
            self.value_opt.zero_grad(); v_loss.backward(); self.value_opt.step()
            if logs: logs['v_loss'] = v_loss.item()
            self._meta_steps += 1
            if self._meta_steps % 500 == 0: self._update_thresholds()
            if self._meta_steps > 5000: self._meta_reliable = True
        return logs or {}
