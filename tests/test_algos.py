"""Smoke tests for the four agents: shapes, action bounds, and that an update runs."""
import numpy as np
import pytest
import torch

from hierrl.algos import DDPG, DEX, SAC, ViSkill, ReplayBuffer

OBS, ACT = 23, 4


def fill(buf, n=300):
    for _ in range(n):
        buf.add(np.random.randn(OBS).astype(np.float32),
                np.random.uniform(-1, 1, ACT).astype(np.float32),
                float(np.random.randn()),
                np.random.randn(OBS).astype(np.float32),
                False)
    return buf


@pytest.mark.parametrize("make", [
    lambda: SAC(OBS, ACT, batch_size=32, warmup_steps=0),
    lambda: DDPG(OBS, ACT, batch_size=32, warmup_steps=0),
    lambda: DEX(OBS, ACT, batch_size=32, warmup_steps=0),
])
def test_act_is_in_bounds(make):
    agent = make()
    a = agent.select_action(np.random.randn(OBS).astype(np.float32))
    assert a.shape == (ACT,)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)
    assert np.all(np.isfinite(a))


@pytest.mark.parametrize("make", [
    lambda: SAC(OBS, ACT, batch_size=32, warmup_steps=0),
    lambda: DDPG(OBS, ACT, batch_size=32, warmup_steps=0),
])
def test_update_runs_and_stays_finite(make):
    agent = make()
    fill(agent.buffer)
    for _ in range(3):
        out = agent.update()
    for p in agent.actor.parameters():
        assert torch.all(torch.isfinite(p))


def test_viskill_exposes_four_sub_policies():
    v = ViSkill(OBS, ACT, n_skills=4, batch_size=32, warmup_steps=0)
    assert len(v.skills) == 4
    a = v.select_action(np.random.randn(OBS).astype(np.float32), env_phase=0)
    assert a.shape == (ACT,) and np.all(np.abs(a) <= 1.0)


def test_train_loop_runs_end_to_end():
    from hierrl.envs.tissue_retract import TissueRetractEnv
    from hierrl.train import train
    env = TissueRetractEnv(max_steps=50, seed=0)
    agent = ViSkill(OBS, ACT, n_skills=4, batch_size=32, warmup_steps=10)
    logs = train("smoke", agent, env, total_steps=120, log_every=60, seed=0, verbose=False)
    assert len(logs["success_rate"]) == 2


@pytest.mark.parametrize("name", ["sac", "ddpg", "dex"])
def test_baseline_train_loops_run_end_to_end(name):
    # Each baseline must survive real updates inside train(), not just construction.
    from hierrl.demos import generate_demos
    from hierrl.envs.tissue_retract import TissueRetractEnv
    from hierrl.train import train
    env = TissueRetractEnv(max_steps=50, seed=0)
    if name == "sac":
        agent = SAC(OBS, ACT, batch_size=32, warmup_steps=10)
    elif name == "ddpg":
        agent = DDPG(OBS, ACT, batch_size=32, warmup_steps=10)
    else:
        agent = DEX(OBS, ACT, batch_size=32, warmup_steps=10, anneal_steps=100)
        for obs_d, act_d, rew_d, nobs_d, dd, _ in generate_demos(env, n_episodes=2):
            agent.add_demo(obs_d, act_d, rew_d, nobs_d, dd)
    logs = train(name, agent, env, total_steps=120, log_every=60, seed=0, verbose=False)
    assert len(logs["success_rate"]) == 2


def test_replay_buffer_roundtrip():
    b = fill(ReplayBuffer(OBS, ACT, capacity=1000), 200)
    assert len(b) == 200
    o, a, r, no, d = b.sample(16)
    assert o.shape == (16, OBS) and a.shape == (16, ACT)
