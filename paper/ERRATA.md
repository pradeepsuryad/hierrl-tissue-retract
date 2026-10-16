# Errata

Corrections to [the paper](hierrl-tissue-retract-aaai.pdf) and [the poster](poster_a0.pdf),
found in October 2026 while re-checking both against the code in this repository. The PDFs
are unchanged; read them together with this list. `scripts/check_errata.py` re-checks every
code-level item below against the committed code, and the re-run results are in
[`results/rerun/`](../results/rerun/).

The paper is a course report (CS 4180/5180, Spring 2026). It was not submitted to or
published by AAAI; the copyright line on its first page comes from the AAAI LaTeX template.

## 1. ViSkill's sub-policies were not gradient-isolated

**Paper.** Each sub-policy has an isolated replay buffer, so "a failed RETRACT does not
penalise APPROACH" (pp. 2–3, Algorithm 1). "The mechanism is gradient isolation" (p. 5).
ViSkill's result "isolates 'gradient mixing' rather than 'reward sparsity' as the dominant
bottleneck" (p. 5 and Conclusion 1). The poster: "each with its own isolated replay buffer".

**Code.** The shared buffer that trains the value network was APPROACH's own replay buffer.
Every transition from every phase went into it with the environment reward, and APPROACH's
SAC updates sampled from it. GRASP, RETRACT and HOLD were isolated; APPROACH was not.

**Re-test.** The fix gives the value network a buffer of its own;
`ViSkill(..., approach_trains_on_all_phases=True)` restores the original. Five seeds per
variant (0, 1, 2, 3, 42), 200,000 steps, everything else identical:

| ViSkill | Final SR | Peak SR | Mean SR | SR, last 50k steps |
|---|---|---|---|---|
| As in the paper (APPROACH shares its buffer) | 64.0% | 93.3% | 66.0% | 68.5% |
| Isolated (fixed) | 52.0% | 96.7% | 61.6% | 51.5% |

Means over the five seeds; per-seed values and ranges are in `results/rerun/summary.md`.

Isolating APPROACH did not help. Over the last 50,000 steps the original was ahead on all
five seeds, although with five runs per variant the difference is not significant (exact
two-sided permutation test, p = 0.063). The claims that gradient isolation explains
ViSkill's result, and that the result identifies gradient mixing as the dominant bottleneck
for flat policies, are withdrawn.

## 2. Table 2 is one run per algorithm, and seed 42 does not reproduce it

Table 2, Figure 1 and the abstract report a single run per algorithm. The seed fixes the
environment's random stream but not the networks' initial weights. The notebook seeds
PyTorch once, at the top, then builds and trains SAC, DDPG, DEX and ViSkill in turn, so each
agent starts from whatever generator state the runs before it left behind. Re-running the
published ViSkill code with seed 42 in a fresh process gives 43.3% final and 93.3% peak
success, not 60.0% and 96.7%.

All four algorithms were re-run on five seeds with the paper's settings
(`scripts/rerun.py`; ranges, per-run values and tests in `results/rerun/summary.md`):

| Algorithm | Final SR | Peak SR | SR, last 50k steps | Mean R |
|---|---|---|---|---|
| SAC | 10.7% | 21.3% | 10.1% | −5.7 |
| DDPG | 30.0% | 40.7% | 27.4% | −38.9 |
| DEX | 22.7% | 57.3% | 17.7% | +16.7 |
| ViSkill, as in the paper | 64.0% | 93.3% | 68.5% | −0.9 |

What changes in the paper's statements:

- **"ViSkill's 60% vs. 10–23% for flat baselines"** (Conclusion 1) becomes 64.0% against
  10.7–30.0% final success for SAC, DDPG and DEX. The advantage is not a single-seed effect:
  on success over the last 50,000 steps and on mean success, every ViSkill run beat every
  SAC, DDPG and DEX run (p = 0.008, the smallest a five-against-five permutation test can
  give). Its cause is not gradient isolation (item 1).
- **"A 3.6× improvement over flat SAC"** (p. 5) is 6.0× on final success over five seeds.
- **The baselines reorder.** DEX (22.7%) finishes ahead of SAC (10.7%), so DEX does not
  have "the lowest final success rate" (p. 5); it still has the highest mean reward.
- **"DEX reaches 50% success at step 15,000—approximately 7× faster than SAC's first
  success at step 110,000"** (p. 5). DEX first reaches 50% at 15–25k steps in four of five
  runs and never in the fifth; SAC's first success comes at 65–115k steps (94k on average).
- **"Once λ(t) reaches 0 at step 80,000, success drops from a 50% peak to 10% final"**
  (p. 5). DEX does decline as the behaviour-cloning weight anneals, but gradually rather
  than at step 80,000: mean success is 37.6% over the first 40,000 steps, 22.7% over
  40–80k, and about 15% once λ is zero.
- **ViSkill "reaches near-peak performance by step 25,000"** holds: it first reaches 80% at
  20–25k steps in all five runs.

## 3. The environment and reward sections do not describe the code

`hierrl/envs/tissue_retract.py` and `hierrl/algos/viskill.py` are extracted verbatim from
the notebook that produced the results.

**a. The tissue has no dynamics.** Eqs. 5–7 (semi-implicit Euler at Δt = 0.01 s, damping
10 N·s/m, point mass 0.05 kg) are not in the code, which has no velocity, mass or time step.
Before the grasp, each anchor moves 5% of the way back to its rest position every step.
After it, each anchor sits at its rest position plus the end effector's displacement scaled
by exp(−10 d), where d is the anchor's rest distance from the grasp target. Force is
50 N/m times the anchors' displacement, averaged over the four anchors. The text's
k_s = 200 N/m is also wrong; Table 1's 50 N/m matches the code.

**b. Phase thresholds.** The text puts APPROACH → GRASP at d_ee < 0.01 m and
GRASP → RETRACT at d_ee < 0.02 m. The code uses 0.06 m for both (Table 1's "phase dist.
threshold").

**c. The environment's +10 grasp bonus (r_G in Eq. 4) never fires.** The poster has it as
"r = +10 on jaw close". The reward is computed before the grasp is registered, and the step
that registers it also moves the phase to RETRACT, so no step is ever rewarded in GRASP
with the grasp made.

**d. ViSkill's +5 grasp bonus never fires either.** `train()` files each transition under
the phase after the step, not the phase in which the action was taken (Algorithm 1, line
8), so the grasp step lands in RETRACT's buffer and GRASP's +5 is never stored. Neither
grasp bonus reaches any learner.

**e. ViSkill's per-phase rewards are not Eq. 4**, although Algorithm 1 (line 7) says they
are. They are defined in `ViSkill._skill_reward`:

| Phase | Reward in the code |
|---|---|
| APPROACH | −0.5 d_ee |
| GRASP | +5 once grasped (see d), otherwise −0.3 d_ee |
| RETRACT | 0.5 d_tissue / 0.15 m |
| HOLD | +1 if d_tissue > 0.09 m, otherwise −0.2 |

Every case except the +5 adds a force penalty of −0.3 log(1 + max(0, F − 5 N)). HOLD is the
binary threshold the paper says v3T replaced. The environment's own reward treats HOLD
exactly like RETRACT; Eq. 4's continuous HOLD row is the v3T reward (item 5), which is not
in this code.

**f. Anchor jitter** is uniform in [−0.02, 0.02] m per axis, not N(0, 0.02² I). The poster
has the same error.

## 4. Table 3: v3T's training success is 56.7%, not 57.0%

A 30-episode success window moves in steps of 3.3 points, so 57.0% cannot occur. 56.7%
(17 of 30) is also the value the 6.7 pp gap implies.

## 5. The generalisation study cannot be re-run here

Table 3, Figures 4 and 5, the v3 and v3T results and Conclusions 3–5 are pre-fix runs from
the paper; not reproducible from this repository. The 26-dimensional goal-aware
observation, the ViSkill-DEX sub-agents, the BC floor, the continuous HOLD reward and the
10-seed held-out evaluation are not in this code.
