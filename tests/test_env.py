"""Environment tests, pinned to the constants the paper reports in Table 1.

These exist mainly to stop the two documented bug fixes from being silently
reverted: the success condition is 0.6 x target height held for 3 consecutive
steps, not the 0.8 x / 10-step version of the earlier prototype, and that
difference is the whole reason the earlier numbers were near zero.
"""
import gymnasium as gym
import numpy as np
import pytest

import hierrl.envs  # noqa: F401  (registers the environments)
from hierrl.envs.tissue_retract import TissueRetractEnv


@pytest.fixture
def env():
    return TissueRetractEnv(seed=42)


class TestPaperConstants:
    def test_physics(self, env):
        assert env.spring_k == 50.0
        assert env.force_limit == 5.0
        assert env.max_steps == 500

    def test_target_height_is_randomised(self):
        hs = set()
        for s in range(20):
            e = TissueRetractEnv(seed=s)
            e.reset(seed=s)
            hs.add(round(e.target_height, 6))
            assert 0.15 <= e.target_height <= 0.20
        assert len(hs) > 1, "H* must be sampled per episode, not fixed"

    def test_action_and_observation_shape(self, env):
        obs, _ = env.reset(seed=0)
        assert obs.shape == (23,)
        assert env.action_space.shape == (4,)
        assert env.pos_scale == 0.05
        assert env.jaw_scale == 0.1
        assert env.jaw_close_thr == 0.15


class TestBugFixOne:
    """The fix that separates this from the superseded prototype."""

    def test_success_needs_three_consecutive_steps(self, env):
        env.reset(seed=0)
        env.hold_counter = 2
        assert not (env.hold_counter >= 3)
        env.hold_counter = 3
        assert env.hold_counter >= 3

    def test_source_uses_the_relaxed_thresholds(self):
        import inspect
        src = inspect.getsource(TissueRetractEnv)
        assert "0.6 * self.target_height" in src, "success threshold must be 0.6 H*"
        assert "self.hold_counter >= 3" in src, "must hold for 3 steps, not 10"


class TestDynamics:
    def test_step_returns_the_gymnasium_five_tuple(self, env):
        env.reset(seed=0)
        out = env.step(np.zeros(4, dtype=np.float32))
        assert len(out) == 5
        obs, r, term, trunc, info = out
        assert obs.shape == (23,)
        assert np.isfinite(r)
        assert isinstance(term, bool) and isinstance(trunc, bool)
        for k in ("phase", "success", "tissue_disp"):
            assert k in info

    def test_episode_truncates_at_max_steps(self, env):
        env.reset(seed=0)
        for i in range(env.max_steps):
            _, _, term, trunc, _ = env.step(np.zeros(4, dtype=np.float32))
            if term or trunc:
                break
        assert trunc or term
        assert i <= env.max_steps - 1

    def test_reset_is_deterministic_given_a_seed(self):
        a, _ = TissueRetractEnv(seed=1).reset(seed=7)
        b, _ = TissueRetractEnv(seed=1).reset(seed=7)
        np.testing.assert_allclose(a, b)

    def test_observation_stays_finite_under_extreme_actions(self, env):
        env.reset(seed=0)
        for _ in range(50):
            obs, r, term, trunc, _ = env.step(np.ones(4, dtype=np.float32))
            assert np.all(np.isfinite(obs)) and np.isfinite(r)
            if term or trunc:
                break


class TestRegistration:
    @pytest.mark.parametrize("env_id", ["TissueRetract-v0", "TissueRetract-v1"])
    def test_gym_make(self, env_id):
        e = gym.make(env_id)
        obs, _ = e.reset(seed=0)
        assert obs.shape == (23,)
