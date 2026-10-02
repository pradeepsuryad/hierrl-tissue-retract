# HierRL-TissueRetract

**Hierarchical Skill Chaining vs. Demonstration-Guided Reinforcement Learning for Surgical Tissue Retraction: A Comparative Study**

Pradeep Surya Dadi — Khoury College of Computer Sciences, Northeastern University
CS 4180/5180 (Reinforcement Learning), Spring 2026 · `dadi.pr@northeastern.edu`

[![tests](https://github.com/pradeepsuryad/hierrl-tissue-retract/actions/workflows/tests.yml/badge.svg)](https://github.com/pradeepsuryad/hierrl-tissue-retract/actions/workflows/tests.yml)

📄 **[Read the paper](paper/hierrl-tissue-retract-aaai.pdf)** · 🖼 [Poster](paper/poster_a0.pdf) · 📓 [Notebook](notebooks/hierrl_v2_final.ipynb) · 📚 [Deep dive](docs/DEEP_DIVE.md)

Autonomous tissue retraction — approach soft tissue, grasp it, lift it to a target
displacement, hold it without tearing — formalised as a continuous-control MDP with a
four-phase internal structure (APPROACH → GRASP → RETRACT → HOLD), and used to compare
four deep RL algorithms on a CPU-only Gymnasium environment with a spring-mass tissue
proxy.

![Training dashboard](figures/fig1_training_dashboard.png)

## Results at 200,000 steps

| Algorithm | Final SR | Peak SR | Mean reward | Mean SR |
|---|---|---|---|---|
| SAC | 16.7% | 16.7% | −4.4 | 5.8% |
| DDPG | 23.3% | 33.3% | −43.7 | 12.5% |
| DEX | 10.0% | 50.0% | **+23.1** | 16.0% |
| **ViSkill** | **60.0%** | **96.7%** | +4.5 | **55.6%** |

Success = tissue displacement ≥ 0.6 H\* held for 3 consecutive steps, on a 30-episode
sliding window. Seed 42, single CPU core.

**Hierarchy wins on sustained performance.** ViSkill reaches near-peak by step 25,000 —
4.4× fewer steps (25k vs 110k) than SAC's first success — through gradient isolation:
each sub-policy receives gradients only from its own phase, so a failed RETRACT cannot
corrupt the APPROACH policy.

**Reward and success measure orthogonal things here.** DEX has the *highest* mean reward
(+23.1) and the *lowest* final success (10%); DDPG reaches 23.3% success with mean reward
−43.7, ten times the force penalty of a successful SAC episode. Its Ornstein-Uhlenbeck
noise is double-edged: correlated exploration is what gets the jaw closed, and the same
correlation then produces sustained upward force during RETRACT. On a real robot that
policy would routinely approach the tear threshold. Any single-metric ranking of these
four would mis-order them.

## Closing the generalisation gap

| Variant | Train SR | Eval SR | Gap |
|---|---|---|---|
| ViSkill-SAC (v2, 23D obs) | 60.0% | 10.0% | 50.0 pp |
| ViSkill-DEX (v3, 26D obs) | 80.0% | 30.0% | 50.0 pp |
| **ViSkill-DEX (v3T, tuned)** | 56.7% | **50.0%** | **6.7 pp** |

Eval is 10 held-out seeds (100–109) with the deterministic greedy policy.

![ViSkill skill timeline](figures/fig3_viskill_skill_timeline.png)

The v3→v3T row reads like a regression — training success *drops* 80% → 57% — and is the
opposite. Eval success rises to 50% and the gap collapses to 6.7 pp. The BC floor
(λ_min = 0.15) stops the HOLD sub-policy exploiting training-seed idiosyncrasies, trading
training success for behaviour that stays near demonstrations computed from geometric
primitives invariant to the per-seed anchor pattern. Lowering training peak performance
was the fastest route to eval robustness.

## What the study concludes

1. **Hierarchy is necessary for contact-rich sequential tasks.** 60% vs 10–23% for flat
   baselines isolates *gradient mixing*, not reward sparsity, as the dominant bottleneck.
2. **Demo quality dominates demonstration-guided RL.** Phase-separated demonstrations
   (jaw closes fully before any translation) take the scripted controller from 0% to 96%
   success, and DEX from 3.3% to 50% peak. No hyperparameter change comes close.
3. **Linear BC annealing to zero causes collapse.** DEX peaks at 50% by step 15,000 and
   falls to 10% once λ(t) hits zero at step 80,000. Flooring λ ≥ 0.15 sustains it.
4. **Observation design is algorithmic design.** The 50 pp → 6.7 pp gap reduction came
   primarily from making the observation goal-aware (23D → 26D). No algorithmic change
   alone produced anything comparable — *no algorithm can compensate for information
   missing from the observation.*
5. **Continuous dense rewards beat binary cliff rewards near decision boundaries.**
   Replacing the binary HOLD reward with min(1, d/H\*) removed a zero-gradient region and
   converted 4 of 6 HOLD failures into successes.

## Ablations

| Ablation | Result |
|---|---|
| Naive demos (all dimensions move together) | 0% scripted → DEX 3.3% final |
| Phase-separated demos (jaw first) | 96% scripted → DEX 50% peak |
| ViSkill without the −0.5 GRASP jaw bias | **0%** — permanently stuck in APPROACH |
| ViSkill with jaw bias | 60% final / 96.7% peak |

## Environment

`TissueRetract-v0` — Gymnasium-compatible, CPU-only, no external physics engine.

| | |
|---|---|
| Observation | 23D (v2) / 26D goal-aware (v3, v3T) |
| Action | 4D — Δx, Δy, Δz (±0.05 m), Δjaw (±0.1 rad) |
| Tissue | 4 anchors, Hookean springs, semi-implicit Euler at Δt = 0.01 s |
| Constants | k = 50 N/m, damping 10 N·s/m, m = 0.05 kg, F_lim = 5 N, tear at 2 F_lim |
| Randomisation | grasp target ~ U([−0.05, 0.05]³), target height H\* ~ U([0.15, 0.20]) m |
| Termination | success (3 consecutive steps), tear, or 500-step truncation |

## Running it

```bash
pip install -e .
pytest tests/ -q                              # 18 tests
python scripts/verify_paper_claims.py         # re-checks conclusion 2
```

`hierrl/` is extracted from `notebooks/hierrl_v2_final.ipynb` — the notebook that
produced the published results — so the package and the paper cannot drift apart. The
environment was checked constant by constant against Table 1: spring constant, force
limit, tear threshold, randomised target height, the 0.6 H\* / 3-consecutive-step
success condition, jaw threshold and scale, translation scale, episode length, and every
per-algorithm learning rate, batch size, buffer and warmup, plus the 80k BC anneal
horizon and the −0.5 GRASP jaw bias. Both documented bug fixes are present and the tests
pin them, so the thresholds that made the earlier prototype score near zero cannot be
reintroduced unnoticed.

`scripts/verify_paper_claims.py` reproduces conclusion 2 directly: the phase-separated
demonstration controller succeeds on **50/50** episodes, and the naive controller that
retracts while the jaw is still closing scores **0/50**. (The paper reports 96% for the
phase-separated controller; the difference is the 0.04 action noise the notebook's demo
generator adds for coverage, which this deterministic check omits.)

### What is not here

- **The v3 / v3T generalisation study.** The 26D goal-aware observation, the BC floor
  (λ_min = 0.15), the continuous HOLD reward, the ViSkill-DEX sub-agents and the
  held-out-seed evaluation are not implemented in this codebase. The generalisation
  table above and conclusions 3 and 4 come from the paper and cannot yet be re-run here.
- **Stored training results.** The notebook is committed without outputs; the figures in
  `figures/` were extracted from a local executed copy. Reproducing the numbers means
  re-running four algorithms for 200,000 steps each.

One paper/code discrepancy found while checking: the paper specifies anchor jitter
ε ~ N(0, 0.02² I), while the code uses `uniform(-0.02, 0.02)`.

## Contents

```
hierrl/     the package: envs/tissue_retract.py, algos/{sac,ddpg,dex,viskill}.py,
            demos.py, train.py - extracted from the notebook
tests/      18 tests pinning the paper's Table 1 constants and both bug fixes
scripts/    verify_paper_claims.py
paper/      the AAAI-format paper and the A0 poster
notebooks/  hierrl_v2_final.ipynb - the original end-to-end run
figures/    training dashboard and ViSkill skill timeline
docs/       DEEP_DIVE.md - cell-by-cell walkthrough of every design decision
```

## Provenance

An earlier prototype repository (`hierrl_tissue_retract`) predated three environment
fixes that are load-bearing for these results — the success threshold (0.8 H\* held 10
steps → 0.6 H\* held 3), the spring constant (5 → 50 N/m) and randomised target height —
and reported near-zero success for every algorithm as a consequence. Its useful parts
have been folded into this repository, which is now the single reference implementation.

## Ethics

Simulation study only. No patient data, no physical robot experiments; tissue dynamics are
a spring-mass proxy. Real deployment would require clinical validation, regulatory
approval, and surgeon-in-the-loop oversight beyond this scope.
