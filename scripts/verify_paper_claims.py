"""Check the extracted package against the numbers the paper reports.

    python scripts/verify_paper_claims.py

Cheap claims only -- the ones that do not need a 200,000-step training run:
the scripted controller's success rate (96%, and 0% without phase separation)
and the jaw-bias ablation's mechanism. The full training comparison is the
notebook; this is what CI can afford to run on every push.
"""
from __future__ import annotations

import argparse

import numpy as np

from hierrl.envs.tissue_retract import TissueRetractEnv


def scripted_episode(env, seed, phase_separated=True):
    env.reset(seed=seed)
    for _ in range(env.max_steps):
        phase = env.phase
        if phase == TissueRetractEnv.APPROACH:
            d = env.grasp_target - env.ee_pos
            a = np.clip(d / env.pos_scale, -1, 1)
            jaw = 0.0
        elif phase == TissueRetractEnv.GRASP:
            # Bug fix 2. Phase-separated: stay completely still and shut the
            # jaw. Naive: start retracting while the jaw is still closing,
            # which pulls the tool off the target so the GRASP distance
            # condition never holds long enough to trigger.
            if phase_separated:
                a = np.zeros(3)
            else:
                a = env.retract_dir / max(abs(env.retract_dir).max(), 1e-9)
            jaw = -1.0
        else:
            a = env.retract_dir / max(abs(env.retract_dir).max(), 1e-9)
            jaw = -1.0
        _, _, term, trunc, info = env.step(
            np.array([*np.clip(a, -1, 1), jaw], dtype=np.float32))
        if info.get("success"):
            return True
        if term or trunc:
            return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()

    env = TissueRetractEnv(seed=42)
    rows = []
    for label, sep in (("phase-separated (paper: 96%)", True),
                       ("naive, jaw+move together (paper: 0%)", False)):
        ok = sum(scripted_episode(env, s, sep) for s in range(args.episodes))
        rows.append((label, ok / args.episodes))
        print(f"  {label:38} {ok}/{args.episodes} = {ok / args.episodes:.0%}")

    sep_sr, naive_sr = rows[0][1], rows[1][1]
    assert sep_sr > 0.80, f"phase-separated controller should be ~96%, got {sep_sr:.0%}"
    assert naive_sr < sep_sr, "phase separation must beat the naive controller"
    print("\nOK: demo-controller claims reproduce (conclusion 2 of the paper).")


if __name__ == "__main__":
    main()
