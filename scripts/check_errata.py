"""Re-check the code-level items in paper/ERRATA.md against the package.

    python scripts/check_errata.py

Each check runs the committed code, prints what it found, and fails loudly if
the code no longer behaves the way the errata describe -- so the errata and
the code cannot drift apart unnoticed. The multi-seed training results are
backed by results/rerun/ instead; they need hours of CPU, not seconds.
"""
from __future__ import annotations

import numpy as np

from hierrl.algos.viskill import ViSkill
from hierrl.envs.tissue_retract import TissueRetractEnv
from hierrl.train import train

OBS, ACT = 23, 4


def goal_seeking_action(env, rng):
    """Head for the grasp target, shut the jaw once within reach, then lift.

    Noisy, so it grasps on most episodes but not deterministically."""
    a = np.zeros(4, dtype=np.float32)
    if env.grasped:
        a[:3] = env.retract_dir
    else:
        a[:3] = np.clip((env.grasp_target - env.ee_pos) / env.pos_scale, -1, 1)
    a[3] = -1.0 if np.linalg.norm(env.grasp_target - env.ee_pos) < env.grasp_radius else 1.0
    return np.clip(a + rng.normal(0, 0.3, 4), -1, 1).astype(np.float32)


def check_buffer_sharing():
    """Erratum 1: in the original code, APPROACH's buffer was the shared one."""
    original = ViSkill(OBS, ACT, approach_trains_on_all_phases=True)
    fixed = ViSkill(OBS, ACT)
    assert original.global_buffer is original.skills[0].buffer
    assert fixed.global_buffer is not fixed.skills[0].buffer
    print("  1. original ViSkill: the value net's buffer IS APPROACH's replay buffer;"
          " fixed: separate")


def check_tissue_is_kinematic(episodes=50):
    """Erratum 3a: no velocity, mass or time step; anchors are placed, not integrated."""
    env = TissueRetractEnv(seed=0)
    rng = np.random.default_rng(0)
    state = [k for k in vars(env) if any(s in k for s in ("vel", "mass", "damp", "dt"))]
    assert not state, f"env now has dynamic state: {state}"
    worst = 0.0
    for _ in range(episodes):
        env.reset()
        for _ in range(env.max_steps):
            _, _, term, trunc, _ = env.step(goal_seeking_action(env, rng))
            if env.grasped:
                falloff = np.exp(-10.0 * np.linalg.norm(env.anchor_rest - env.grasp_target, axis=1))
                placed = env.anchor_rest + (env.ee_pos - env.grasp_point) * falloff[:, None]
                worst = max(worst, float(np.abs(env.anchor_pos - placed).max()))
            if term or trunc:
                break
    assert worst < 1e-6, f"anchors deviate from the kinematic placement by {worst}"
    assert env.spring_k == 50.0
    print(f"  3a. tissue is kinematic: grasped anchors match rest + displacement x exp(-10 d)"
          f" to {worst:.0e} m; k = {env.spring_k:g} N/m (paper text: 200)")


def check_phase_thresholds():
    """Erratum 3b: both distance thresholds are 0.06 m, not 0.01 / 0.02 m."""
    env = TissueRetractEnv(seed=0)
    env.reset(seed=0)
    env.ee_pos = (env.grasp_target + np.array([0.05, 0, 0], dtype=np.float32)).astype(np.float32)
    env.step(np.zeros(4, dtype=np.float32))
    assert env.phase == TissueRetractEnv.GRASP, "0.05 m from the target should already be GRASP"
    for _ in range(10):                                     # shut the jaw without moving
        env.step(np.array([0, 0, 0, -1], dtype=np.float32))
    assert env.phase >= TissueRetractEnv.RETRACT, "jaw shut 0.05 m away should already grasp"
    print(f"  3b. APPROACH->GRASP and GRASP->RETRACT both trigger at 0.05 m"
          f" (threshold {env.grasp_radius} m; paper text: 0.01 and 0.02 m)")


def check_env_grasp_bonus(episodes=300):
    """Erratum 3c: the environment's +10 grasp bonus never reaches a reward."""
    env = TissueRetractEnv(seed=1)
    rng = np.random.default_rng(1)
    grasps, best = 0, -np.inf
    for _ in range(episodes):
        env.reset()
        for _ in range(env.max_steps):
            was = env.grasped
            _, r, term, trunc, info = env.step(goal_seeking_action(env, rng))
            best = max(best, r)
            grasps += info["grasped"] and not was
            if term or trunc:
                break
    assert grasps > 0.5 * episodes, f"probe controller grasped only {grasps} times"
    assert best < 1.0, f"a reward of {best} means the grasp bonus can fire"
    print(f"  3c. env grasp bonus: {grasps} grasps, largest reward on any step {best:.2f}"
          f" (bonus would be +10)")


def check_viskill_grasp_bonus(steps=20_000):
    """Erratum 3d: train() files each transition under the phase *after* the step,
    so the grasp step lands in RETRACT's buffer and GRASP's +5 is never stored."""
    env = TissueRetractEnv(seed=2)
    rng = np.random.default_rng(2)
    agent = ViSkill(OBS, ACT)
    agent.select_action = lambda obs, env_phase=None, deterministic=False: goal_seeking_action(env, rng)
    agent.update = lambda env_phase=0: {}                   # the policy is irrelevant here
    stored, store = [], agent.store

    def recording_store(obs, action, reward, next_obs, done, env_phase, info):
        skill = min(int(env_phase), agent.n_skills - 1)
        stored.append((skill, agent._skill_reward(info, skill), info["grasped"]))
        store(obs, action, reward, next_obs, done, env_phase, info)

    agent.store = recording_store
    train("errata", agent, env, total_steps=steps, log_every=steps, seed=2, verbose=False)
    grasp_rewards = [r for skill, r, _ in stored if skill == TissueRetractEnv.GRASP]
    grasped_skills = {skill for skill, _, grasped in stored if grasped}
    assert grasp_rewards and max(grasp_rewards) < 5.0, "GRASP's +5 sub-reward was stored"
    assert TissueRetractEnv.GRASP not in grasped_skills
    print(f"  3d. ViSkill: {len(grasp_rewards)} GRASP transitions stored, largest sub-reward"
          f" {max(grasp_rewards):.2f} (bonus would be +5); grasped transitions filed under"
          f" phases {sorted(grasped_skills)}")


def check_viskill_rewards():
    """Erratum 3e: ViSkill's per-phase rewards, as tabulated in the errata."""
    r = ViSkill._skill_reward
    fp = -0.3 * np.log(1 + (8.0 - 5.0))                     # force penalty at F = 8 N
    cases = [
        (0, dict(dist_grasp=0.1), -0.05),                   # APPROACH: -0.5 d_ee
        (1, dict(dist_grasp=0.1), -0.03),                   # GRASP, not grasped: -0.3 d_ee
        (1, dict(dist_grasp=0.1, grasped=True, force=8.0), 5.0),   # +5, no force penalty
        (2, dict(tissue_disp=0.15), 0.5),                   # RETRACT: 0.5 d_tissue / 0.15
        (3, dict(tissue_disp=0.10), 1.0),                   # HOLD: binary at 0.09 m
        (3, dict(tissue_disp=0.08), -0.2),
        (0, dict(dist_grasp=0.0, force=8.0), fp),           # -0.3 log(1 + max(0, F - 5))
    ]
    for phase, info, want in cases:
        got = r(info, phase)
        assert abs(got - want) < 1e-9, f"phase {phase} {info}: {got} != {want}"
    print(f"  3e. ViSkill per-phase rewards match the errata table ({len(cases)} cases)")


def check_anchor_jitter(resets=2000):
    """Erratum 3f: anchor jitter is uniform in [-0.02, 0.02], not N(0, 0.02^2)."""
    env = TissueRetractEnv(seed=3)
    offsets = np.array([[-0.05, -0.05, 0], [0.05, -0.05, 0], [-0.05, 0.05, 0], [0.05, 0.05, 0]])
    noise = []
    for _ in range(resets):
        env.reset()
        noise.append(env.anchor_rest - env.grasp_target - offsets)
    noise = np.abs(np.concatenate(noise))
    assert noise.max() <= 0.02 + 1e-6, "jitter exceeds 0.02, so it is no longer uniform(+-0.02)"
    print(f"  3f. anchor jitter: max |noise| over {resets} resets = {noise.max():.4f} m"
          f" (N(0, 0.02^2) would exceed 0.02 m about a third of the time)")


def main():
    print("Checking paper/ERRATA.md against the code:")
    check_buffer_sharing()
    check_tissue_is_kinematic()
    check_phase_thresholds()
    check_env_grasp_bonus()
    check_viskill_grasp_bonus()
    check_viskill_rewards()
    check_anchor_jitter()
    print("\nOK: every code-level erratum still describes the code.")


if __name__ == "__main__":
    main()
