"""Summarise results/rerun/*.json into results/rerun/summary.md and fig_rerun_success.png.

    python scripts/summarize_rerun.py [results/rerun]

The JSON files come from scripts/rerun.py: five seeds each of SAC, DDPG, DEX
and ViSkill with and without the paper's buffer sharing.
"""
import glob
import itertools
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent / "results" / "rerun")
ORDER = [("sac", "SAC"), ("ddpg", "DDPG"), ("dex", "DEX"),
         ("original", "ViSkill, as in the paper"), ("fixed", "ViSkill, isolated (fixed)")]
runs = {}
for f in sorted(glob.glob(f"{RES}/*_seed*.json")):
    r = json.load(open(f))
    runs.setdefault(r["variant"], []).append(r)
for v in runs:
    runs[v].sort(key=lambda r: r["seed"])


def last50(r):
    return float(np.mean(r["logs"]["success_rate"][-10:]))      # last 10 logs x 5,000 steps


def first_success(r):
    """First logged step whose 30-episode window contains a success (None if never)."""
    for step, sr in zip(r["logs"]["step"], r["logs"]["success_rate"]):
        if sr > 0:
            return step
    return None


def fmt(xs, pct=True):
    m, lo, hi = st.mean(xs), min(xs), max(xs)
    return f"{m:.1%} ({lo:.1%}–{hi:.1%})" if pct else f"{m:+.1f} ({lo:+.1f} to {hi:+.1f})"


def fmt_steps(xs):
    got = [x for x in xs if x is not None]
    never = len(xs) - len(got)
    if not got:
        return "never"
    s = f"{st.mean(got)/1000:.0f}k ({min(got)/1000:.0f}k–{max(got)/1000:.0f}k)"
    return s + (f", never in {never}" if never else "")


def perm_p(a, b):
    """Exact two-sided permutation test on the difference of means (unpaired)."""
    a, b = np.asarray(a), np.asarray(b)
    pooled, n, obs = np.concatenate([a, b]), len(a), abs(a.mean() - b.mean())
    hits = total = 0
    for idx in itertools.combinations(range(len(pooled)), n):
        mask = np.zeros(len(pooled), bool); mask[list(idx)] = True
        hits += abs(pooled[mask].mean() - pooled[~mask].mean()) >= obs - 1e-12
        total += 1
    return hits / total


present = [(v, name) for v, name in ORDER if v in runs]
seeds = sorted({r["seed"] for v, _ in present for r in runs[v]})
L = ["# Re-run: five seeds per algorithm", "",
     "Same code path and settings as the notebook (200,000 steps, log every 5,000 steps, success = 30-episode",
     "sliding window of training episodes), one CPU thread per run, seeds " + ", ".join(map(str, seeds)) + ".",
     "`ViSkill, as in the paper` is the original ViSkill, whose APPROACH buffer also receives every phase's",
     "transitions (`approach_trains_on_all_phases=True`); `ViSkill, isolated` gives the value net its own buffer,",
     "so each sub-policy trains only on its own phase. A seed fixes the environment's random stream, not the",
     "networks' initial weights, so runs are independent samples rather than seed-for-seed reproductions.", "",
     "Mean over seeds, with the min–max range in brackets. Final SR is a single 30-episode window, so it is",
     "noisy; \"SR, last 50k steps\" averages the last 10 logged windows. \"First success\" is the first logged",
     "step whose window contains a successful episode.", "",
     "| Algorithm | Runs | Final SR | Peak SR | Mean SR | SR, last 50k steps | First success | Mean R |",
     "|---|---|---|---|---|---|---|---|"]
for v, name in present:
    rs = runs[v]
    L.append(f"| {name} | {len(rs)} | {fmt([r['final_sr'] for r in rs])} | {fmt([r['peak_sr'] for r in rs])} | "
             f"{fmt([r['mean_sr'] for r in rs])} | {fmt([last50(r) for r in rs])} | "
             f"{fmt_steps([first_success(r) for r in rs])} | {fmt([r['mean_reward'] for r in rs], pct=False)} |")

if len(present) > 1:
    L += ["", "Exact two-sided permutation tests on the per-run values (unpaired; with five runs per arm the",
          "smallest possible p is 0.008):", "",
          "| Comparison | SR, last 50k steps | Final SR | Mean SR |", "|---|---|---|---|"]
    pairs = [("original", "fixed")] + [(vis, b) for vis in ("original", "fixed") for b in ("sac", "ddpg", "dex")]
    names = dict(ORDER)
    for a, b in pairs:
        if a in runs and b in runs:
            ra, rb = runs[a], runs[b]
            L.append(f"| {names[a]} vs {names[b]} | {perm_p([last50(r) for r in ra], [last50(r) for r in rb]):.3f} | "
                     f"{perm_p([r['final_sr'] for r in ra], [r['final_sr'] for r in rb]):.3f} | "
                     f"{perm_p([r['mean_sr'] for r in ra], [r['mean_sr'] for r in rb]):.3f} |")



def first_at(r, thr):
    """First logged step whose window reaches thr (None if never)."""
    for step, sr in zip(r["logs"]["step"], r["logs"]["success_rate"]):
        if sr >= thr - 1e-9:
            return step
    return None


def window_mean(r, lo, hi):
    steps, sr = np.array(r["logs"]["step"]), np.array(r["logs"]["success_rate"])
    return float(sr[(steps > lo) & (steps <= hi)].mean())


if "dex" in runs or "original" in runs:
    L += ["", "When things happen (mean over runs, min–max in brackets):", "",
          "| Measure | Runs |", "|---|---|"]
    if "sac" in runs:
        L.append(f"| SAC, first logged success | {fmt_steps([first_success(r) for r in runs['sac']])} |")
    if "dex" in runs:
        L.append(f"| DEX, first window at 50% or more | {fmt_steps([first_at(r, 0.5) for r in runs['dex']])} |")
    for v in ("original", "fixed"):
        if v in runs:
            L.append(f"| {dict(ORDER)[v]}, first window at 80% or more | "
                     f"{fmt_steps([first_at(r, 0.8) for r in runs[v]])} |")
    if "dex" in runs:
        L += ["", "DEX success as the behaviour-cloning weight anneals (λ falls linearly from 1 at step 0 to 0 at",
              "step 80,000 and stays there): mean success rate over each span of training.", "",
              "| Steps | λ over the span | Mean SR |", "|---|---|---|"]
        for lo, hi, lam in ((0, 40_000, "1 → 0.5"), (40_000, 80_000, "0.5 → 0"),
                            (80_000, 120_000, "0"), (120_000, 200_000, "0")):
            L.append(f"| {lo//1000}k–{hi//1000}k | {lam} | {fmt([window_mean(r, lo, hi) for r in runs['dex']])} |")

L += ["", "Per run:", "",
      "| Algorithm | Seed | Final SR | Peak SR | Mean SR | SR, last 50k | First success | Mean R | Wall time |",
      "|---|---|---|---|---|---|---|---|---|"]
for v, name in present:
    for r in runs[v]:
        fs = first_success(r)
        L.append(f"| {name} | {r['seed']} | {r['final_sr']:.1%} | {r['peak_sr']:.1%} | {r['mean_sr']:.1%} | "
                 f"{last50(r):.1%} | {'never' if fs is None else f'{fs/1000:.0f}k'} | {r['mean_reward']:+.1f} | "
                 f"{r['wall_s']/60:.0f} min |")
any_r = runs[present[0][0]][0]
L += ["", f"Environment: Python {any_r['python']}, torch {any_r['torch']} (CPU). Raw logs: "
      "`<algorithm>_seed<seed>.json` (ViSkill: `viskill_<original|fixed>_seed<seed>.json`). Wall times overlap: the runs",
      "shared one machine.", ""]
rebuilt = sorted({dict(ORDER)[v] for v, _ in present for r in runs[v] if r.get("rebuilt_from_log")})
if rebuilt:
    L += [f"The {', '.join(rebuilt)} files were rebuilt from the runs' progress logs after the original JSON files",
          "were lost; `progress/` holds the lines they were read from. Success rates are exact, episode rewards are",
          "rounded to 0.01 and actor losses to 4 decimal places, and critic losses were not printed, so they are empty.", ""]
open(f"{RES}/summary.md", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"sac": "#7f8c8d", "ddpg": "#27ae60", "dex": "#8e44ad", "original": "#c0392b", "fixed": "#2471a3"}
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for v, name in present:
        rs = runs[v]
        steps = np.array(rs[0]["logs"]["step"])
        sr = np.array([r["logs"]["success_rate"] for r in rs])
        ax.plot(steps, sr.mean(0), color=colours[v], label=f"{name} (mean of {len(rs)})")
        ax.fill_between(steps, sr.min(0), sr.max(0), color=colours[v], alpha=0.12)
    ax.set_xlabel("training steps"); ax.set_ylabel("success rate (30-episode window)")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, frameon=False)
    ax.set_title("Five seeds per algorithm: mean and min–max")
    fig.tight_layout(); fig.savefig(f"{RES}/fig_rerun_success.png", dpi=150)
    print("figure written")
except ImportError:
    print("matplotlib missing")
