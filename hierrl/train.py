"""Training loop and the smoothing helper used for the reported curves."""
from collections import deque

import random
import numpy as np
from tqdm.auto import tqdm

def train(name, agent, env, total_steps=30000, log_every=2000, seed=42, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    logs = {'step': [], 'ep_reward': [], 'success_rate': [], 'actor_loss': [], 'critic_loss': []}
    obs, _      = env.reset(seed=seed)
    ep_reward   = 0.0
    recent_r    = deque(maxlen=30)
    recent_succ = deque(maxlen=30)
    start       = time.time()
    pbar = tqdm(range(1, total_steps+1), desc=f'{name:<8}', unit='step',
                colour='blue', dynamic_ncols=True)
    for step in pbar:
        if isinstance(agent, ViSkill):
            action = agent.select_action(obs, env_phase=env.phase)
        else:
            action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if isinstance(agent, ViSkill):
            agent.store(obs, action, reward, next_obs, done, env.phase, info)
        elif isinstance(agent, DEX):
            agent.sac.buffer.add(obs, action, reward, next_obs, float(done))
        else:
            agent.buffer.add(obs, action, reward, next_obs, float(done))
        if isinstance(agent, ViSkill): info_log = agent.update(env_phase=env.phase)
        else: info_log = agent.update()
        ep_reward += reward; obs = next_obs
        if done:
            recent_r.append(ep_reward); recent_succ.append(float(info.get('success', False)))
            obs, _ = env.reset(); ep_reward = 0.0
            if isinstance(agent, DDPG): agent.noise.reset()
        if step % log_every == 0:
            mr = float(np.mean(recent_r))    if recent_r    else 0.0
            sr = float(np.mean(recent_succ)) if recent_succ else 0.0
            al = info_log.get('actor_loss', 0.0)  if info_log else 0.0
            cl = info_log.get('critic_loss', 0.0) if info_log else 0.0
            logs['step'].append(step); logs['ep_reward'].append(mr)
            logs['success_rate'].append(sr); logs['actor_loss'].append(al)
            logs['critic_loss'].append(cl)
            pbar.set_postfix({'reward': f'{mr:>7.2f}', 'success': f'{sr:.1%}',
                              'a_loss': f'{al:.4f}' if al else 'warmup'}, refresh=False)
            if verbose:
                tqdm.write(f'[{name}] step {step:>6} | reward {mr:>7.2f} | success {sr:.1%} | {time.time()-start:.0f}s')
    pbar.close()
    return logs

def smooth(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode='valid') if len(x) >= w else x
