"""Scripted demonstration controller (96% success with phase separation)."""
import random
import numpy as np
from tqdm.auto import tqdm

from .envs.tissue_retract import TissueRetractEnv

def generate_demos(env, n_episodes=150, noise_scale=0.04):
    """
    BUG FIX 2: explicit jaw-close wait before retract.
    Old version mixed jaw close + movement, causing GRASP condition to never trigger.
    """
    demos, rng, success = [], np.random.default_rng(0), 0

    for ep in tqdm(range(n_episodes), desc='generating demos', unit='ep', colour='green'):
        obs, _ = env.reset()
        done   = False

        while not done:
            phase = env.phase

            if phase == TissueRetractEnv.APPROACH:
                # move toward grasp target, jaw stays open
                delta = env.grasp_target - env.ee_pos
                dist  = np.linalg.norm(delta) + 1e-9
                delta = delta / dist
                action = np.array([delta[0]*0.9, delta[1]*0.9, delta[2]*0.9, 0.3],
                                  dtype=np.float32)

            elif phase == TissueRetractEnv.GRASP:
                # BUG FIX 2: stay completely still, slam jaw shut
                # don't mix movement with jaw close — it prevented GRASP condition
                if env.jaw_angle > env.jaw_close_thr + 0.05:
                    action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
                else:
                    # jaw is closed, tiny nudge to confirm grasped flag
                    action = np.array([0.0, 0.0, 0.01, -1.0], dtype=np.float32)

            else:  # RETRACT or HOLD
                rd = env.retract_dir
                action = np.array([rd[0]*0.3, rd[1]*0.3, rd[2]*1.0, 0.0],
                                  dtype=np.float32)

            # small noise for generalization (less noise than before for cleaner demos)
            action = np.clip(action + rng.normal(0, noise_scale, 4).astype(np.float32), -1.0, 1.0)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            demos.append((obs.copy(), action.copy(), reward, next_obs.copy(), float(done), info))
            obs = next_obs

        if info.get('success', False): success += 1

    sr = success / n_episodes
    print(f'  demo gen complete — {len(demos)} transitions | scripted success: {sr:.1%}')
    return demos
