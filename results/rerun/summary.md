# Re-run: five seeds per algorithm

Same code path and settings as the notebook (200,000 steps, log every 5,000 steps, success = 30-episode
sliding window of training episodes), one CPU thread per run, seeds 0, 1, 2, 3, 42.
`ViSkill, as in the paper` is the original ViSkill, whose APPROACH buffer also receives every phase's
transitions (`approach_trains_on_all_phases=True`); `ViSkill, isolated` gives the value net its own buffer,
so each sub-policy trains only on its own phase. A seed fixes the environment's random stream, not the
networks' initial weights, so runs are independent samples rather than seed-for-seed reproductions.

Mean over seeds, with the min–max range in brackets. Final SR is a single 30-episode window, so it is
noisy; "SR, last 50k steps" averages the last 10 logged windows. "First success" is the first logged
step whose window contains a successful episode.

| Algorithm | Runs | Final SR | Peak SR | Mean SR | SR, last 50k steps | First success | Mean R |
|---|---|---|---|---|---|---|---|
| SAC | 5 | 10.7% (0.0%–23.3%) | 21.3% (10.0%–30.0%) | 4.8% (2.2%–7.4%) | 10.1% (3.7%–19.7%) | 94k (65k–115k) | -5.7 (-9.3 to +3.2) |
| DDPG | 5 | 30.0% (6.7%–40.0%) | 40.7% (36.7%–50.0%) | 14.6% (7.2%–24.7%) | 27.4% (18.3%–38.0%) | 32k (10k–70k) | -38.9 (-46.7 to -29.3) |
| DEX | 5 | 22.7% (10.0%–46.7%) | 57.3% (46.7%–70.0%) | 21.4% (14.7%–30.5%) | 17.7% (9.7%–36.3%) | 5k (5k–5k) | +16.7 (+13.3 to +22.3) |
| ViSkill, as in the paper | 5 | 64.0% (43.3%–86.7%) | 93.3% (83.3%–100.0%) | 66.0% (54.9%–78.2%) | 68.5% (56.0%–91.0%) | 10k (5k–15k) | -0.9 (-9.6 to +3.3) |
| ViSkill, isolated (fixed) | 5 | 52.0% (16.7%–66.7%) | 96.7% (93.3%–100.0%) | 61.6% (52.9%–69.4%) | 51.5% (37.3%–63.3%) | 13k (10k–15k) | +1.4 (-3.2 to +9.9) |

Exact two-sided permutation tests on the per-run values (unpaired; with five runs per arm the
smallest possible p is 0.008):

| Comparison | SR, last 50k steps | Final SR | Mean SR |
|---|---|---|---|
| ViSkill, as in the paper vs ViSkill, isolated (fixed) | 0.063 | 0.397 | 0.413 |
| ViSkill, as in the paper vs SAC | 0.008 | 0.008 | 0.008 |
| ViSkill, as in the paper vs DDPG | 0.008 | 0.008 | 0.008 |
| ViSkill, as in the paper vs DEX | 0.008 | 0.016 | 0.008 |
| ViSkill, isolated (fixed) vs SAC | 0.008 | 0.024 | 0.008 |
| ViSkill, isolated (fixed) vs DDPG | 0.016 | 0.079 | 0.008 |
| ViSkill, isolated (fixed) vs DEX | 0.008 | 0.040 | 0.008 |

When things happen (mean over runs, min–max in brackets):

| Measure | Runs |
|---|---|
| SAC, first logged success | 94k (65k–115k) |
| DEX, first window at 50% or more | 19k (15k–25k), never in 1 |
| ViSkill, as in the paper, first window at 80% or more | 22k (20k–25k) |
| ViSkill, isolated (fixed), first window at 80% or more | 23k (20k–25k) |

DEX success as the behaviour-cloning weight anneals (λ falls linearly from 1 at step 0 to 0 at
step 80,000 and stays there): mean success rate over each span of training.

| Steps | λ over the span | Mean SR |
|---|---|---|
| 0k–40k | 1 → 0.5 | 37.6% (29.9%–47.7%) |
| 40k–80k | 0.5 → 0 | 22.7% (17.5%–32.1%) |
| 80k–120k | 0 | 15.3% (7.1%–24.6%) |
| 120k–200k | 0 | 15.8% (8.3%–26.9%) |

Per run:

| Algorithm | Seed | Final SR | Peak SR | Mean SR | SR, last 50k | First success | Mean R | Wall time |
|---|---|---|---|---|---|---|---|---|
| SAC | 0 | 3.3% | 10.0% | 2.2% | 3.7% | 80k | +3.2 | 149 min |
| SAC | 1 | 0.0% | 23.3% | 4.2% | 3.7% | 110k | -7.0 | 150 min |
| SAC | 2 | 16.7% | 23.3% | 7.4% | 15.7% | 65k | -6.8 | 150 min |
| SAC | 3 | 23.3% | 30.0% | 6.6% | 19.7% | 115k | -8.7 | 153 min |
| SAC | 42 | 10.0% | 20.0% | 3.6% | 7.7% | 100k | -9.3 | 156 min |
| DDPG | 0 | 36.7% | 40.0% | 16.2% | 32.0% | 10k | -35.4 | 125 min |
| DDPG | 1 | 6.7% | 36.7% | 7.2% | 20.0% | 70k | -46.4 | 126 min |
| DDPG | 2 | 33.3% | 50.0% | 24.7% | 38.0% | 20k | -29.3 | 124 min |
| DDPG | 3 | 40.0% | 40.0% | 8.1% | 18.3% | 25k | -46.7 | 124 min |
| DDPG | 42 | 33.3% | 36.7% | 16.9% | 28.7% | 35k | -36.5 | 125 min |
| DEX | 0 | 10.0% | 66.7% | 22.3% | 13.7% | 5k | +17.4 | 154 min |
| DEX | 1 | 46.7% | 53.3% | 30.5% | 36.3% | 5k | +22.3 | 153 min |
| DEX | 2 | 20.0% | 46.7% | 19.7% | 13.7% | 5k | +15.5 | 157 min |
| DEX | 3 | 20.0% | 70.0% | 14.7% | 9.7% | 5k | +14.8 | 153 min |
| DEX | 42 | 16.7% | 50.0% | 19.9% | 15.3% | 5k | +13.3 | 152 min |
| ViSkill, as in the paper | 0 | 60.0% | 83.3% | 54.9% | 56.0% | 10k | +3.3 | 83 min |
| ViSkill, as in the paper | 1 | 86.7% | 100.0% | 78.2% | 91.0% | 5k | +2.5 | 85 min |
| ViSkill, as in the paper | 2 | 73.3% | 90.0% | 66.8% | 69.3% | 15k | -9.6 | 79 min |
| ViSkill, as in the paper | 3 | 56.7% | 100.0% | 72.4% | 68.7% | 10k | -1.9 | 81 min |
| ViSkill, as in the paper | 42 | 43.3% | 93.3% | 57.7% | 57.7% | 10k | +1.1 | 88 min |
| ViSkill, isolated (fixed) | 0 | 16.7% | 93.3% | 65.8% | 54.3% | 15k | -3.1 | 86 min |
| ViSkill, isolated (fixed) | 1 | 66.7% | 100.0% | 54.3% | 37.3% | 10k | -3.2 | 78 min |
| ViSkill, isolated (fixed) | 2 | 60.0% | 100.0% | 69.4% | 63.3% | 10k | +9.9 | 88 min |
| ViSkill, isolated (fixed) | 3 | 56.7% | 96.7% | 65.4% | 61.0% | 15k | +5.9 | 82 min |
| ViSkill, isolated (fixed) | 42 | 60.0% | 93.3% | 52.9% | 41.3% | 15k | -2.7 | 80 min |

Environment: Python 3.11.15, torch 2.14.1+cpu (CPU). Raw logs: `<algorithm>_seed<seed>.json` (ViSkill: `viskill_<original|fixed>_seed<seed>.json`). Wall times overlap: the runs
shared one machine.

The DDPG, DEX, SAC files were rebuilt from the runs' progress logs after the original JSON files
were lost; `progress/` holds the lines they were read from. Success rates are exact, episode rewards are
rounded to 0.01 and actor losses to 4 decimal places, and critic losses were not printed, so they are empty.
