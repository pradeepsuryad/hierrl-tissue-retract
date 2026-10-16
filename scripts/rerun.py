"""One 200,000-step training run with the notebook's settings, saved as JSON.

    python scripts/rerun.py viskill-original 42 results/rerun/viskill_original_seed42.json

ALGO is sac, ddpg, dex, viskill-original or viskill-fixed. viskill-original
restores the paper's buffer sharing (approach_trains_on_all_phases=True); see
paper/ERRATA.md, item 1. results/rerun/ holds five seeds (0, 1, 2, 3, 42) of
each, one CPU thread per run; scripts/summarize_rerun.py turns them into
results/rerun/summary.md. A run takes 1.5-2.5 hours on one CPU core.

The seed fixes the environment's random stream. The networks are built before
train() seeds PyTorch, so a run is an independent sample, not a bit-for-bit
reproduction of an earlier one with the same seed.
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import torch

from hierrl.algos import DDPG, DEX, SAC, ViSkill
from hierrl.demos import generate_demos
from hierrl.envs.tissue_retract import TissueRetractEnv
from hierrl.train import train

TOTAL_STEPS, LOG_EVERY, N_DEMO_EPISODES = 200_000, 5_000, 150     # notebook cell 14
VARIANT = {"sac": "sac", "ddpg": "ddpg", "dex": "dex",
           "viskill-original": "original", "viskill-fixed": "fixed"}


def build(algo, env):
    if algo == "sac":                                                # notebook cell 16
        return SAC(23, 4, lr=3e-4, gamma=0.99, tau=0.005, batch_size=256, warmup_steps=1000)
    if algo == "ddpg":                                               # notebook cell 17
        return DDPG(23, 4, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005,
                    batch_size=128, warmup_steps=1000)
    if algo == "dex":                                                # notebook cell 18
        demos = generate_demos(env, n_episodes=N_DEMO_EPISODES)
        agent = DEX(23, 4, lambda_bc_start=1.0, lambda_bc_end=0.0, anneal_steps=int(TOTAL_STEPS * 0.4),
                    lr=3e-4, gamma=0.99, tau=0.005, batch_size=256, warmup_steps=200)
        for obs, act, rew, next_obs, done, _ in demos:
            agent.add_demo(obs, act, rew, next_obs, done)
        return agent
    return ViSkill(23, 4, n_skills=4, lr=3e-4, gamma=0.99, tau=0.005, batch_size=128,  # cell 19
                   warmup_steps=500, approach_trains_on_all_phases=(algo == "viskill-original"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("algo", choices=sorted(VARIANT))
    ap.add_argument("seed", type=int)
    ap.add_argument("out")
    args = ap.parse_args()

    torch.set_num_threads(1)
    env = TissueRetractEnv(multidirectional=False, max_steps=500, seed=args.seed)
    agent = build(args.algo, env)
    name = "ViSkill" if args.algo.startswith("viskill") else args.algo.upper()
    t0 = time.time()
    logs = train(name, agent, env, total_steps=TOTAL_STEPS, log_every=LOG_EVERY, seed=args.seed, verbose=False)
    sr, rr = logs["success_rate"], logs["ep_reward"]
    rec = {"variant": VARIANT[args.algo], "seed": args.seed, "total_steps": TOTAL_STEPS, "log_every": LOG_EVERY,
           "final_sr": sr[-1], "peak_sr": max(sr), "mean_sr": float(np.mean(sr)), "mean_reward": float(np.mean(rr)),
           "wall_s": round(time.time() - t0, 1), "torch": torch.__version__, "python": platform.python_version(),
           "logs": {k: [float(x) for x in v] for k, v in logs.items()}}
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"{args.algo} seed {args.seed}: final {sr[-1]:.1%} peak {max(sr):.1%} ({rec['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
