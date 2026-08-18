"""TissueRetract-v0 / v1 -- the environment used in the paper.

Extracted verbatim from notebooks/hierrl_v2_final.ipynb so the package and the
notebook cannot drift apart. Both documented bug fixes are in place: the
success condition is 0.6 x target height held for 3 consecutive steps (was
0.8 x, 10 steps), and the demo controller closes the jaw before translating.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TissueRetractEnv(gym.Env):
    APPROACH = 0
    GRASP    = 1
    RETRACT  = 2
    HOLD     = 3

    def __init__(self, multidirectional=False, max_steps=500, seed=None):
        super().__init__()
        self.multidirectional = multidirectional
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self.n_anchors     = 4
        self.spring_k      = 50.0
        self.force_limit   = 5.0
        self.grasp_radius  = 0.06
        self.jaw_close_thr = 0.15
        self.pos_scale = 0.05
        self.jaw_scale = 0.1
        self.obs_dim = 3 + 1 + 7 + self.n_anchors * 3
        self.act_dim = 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(low=-1.0,    high=1.0,    shape=(self.act_dim,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        jitter = lambda s: self._rng.uniform(-s, s, 3).astype(np.float32)
        self.grasp_target  = np.array([0.0, 0.0, 0.0], dtype=np.float32) + jitter(0.05)
        self.target_height = float(0.15 + self._rng.uniform(0, 0.05))
        if self.multidirectional:
            theta = self._rng.uniform(0, np.pi/3)
            phi   = self._rng.uniform(0, 2*np.pi)
            self.retract_dir = np.array([np.sin(theta)*np.cos(phi),
                                         np.sin(theta)*np.sin(phi),
                                         np.cos(theta)], dtype=np.float32)
        else:
            self.retract_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.ee_pos    = self.grasp_target + np.array([0.0, 0.0, 0.2], dtype=np.float32) + jitter(0.05)
        self.jaw_angle = np.float32(0.6)
        self.joints    = np.zeros(7, dtype=np.float32)
        offsets = np.array([[-0.05,-0.05,0],[0.05,-0.05,0],[-0.05,0.05,0],[0.05,0.05,0]], dtype=np.float32)
        noise = self._rng.uniform(-0.02, 0.02, (4, 3)).astype(np.float32)
        self.anchor_rest = self.grasp_target[None, :] + offsets + noise
        self.anchor_pos  = self.anchor_rest.copy()
        self.phase = self.APPROACH
        self.grasped = False
        self.grasp_point  = None
        self.hold_counter = 0
        self.step_count   = 0
        self.total_force  = 0.0
        self.success      = False
        # trajectory recording for visualizer
        self._traj_ee     = []
        self._traj_anchors= []
        self._traj_jaw    = []
        self._traj_phase  = []
        self._traj_disp   = []
        return self._obs(), {}

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)
        delta_pos = action[:3] * self.pos_scale
        delta_jaw = action[3]  * self.jaw_scale
        self.ee_pos    = np.clip(self.ee_pos + delta_pos, -1.0, 1.0)
        self.jaw_angle = float(np.clip(self.jaw_angle + delta_jaw, 0.0, 1.0))
        jd = np.zeros(7, dtype=np.float32)
        jd[:3] = delta_pos * 0.5
        self.joints = np.clip(self.joints + jd, -np.pi, np.pi)
        self._deform_tissue()
        dist_grasp  = float(np.linalg.norm(self.ee_pos - self.grasp_target))
        tissue_disp = float(np.dot(
            self.anchor_pos.mean(axis=0) - self.anchor_rest.mean(axis=0),
            self.retract_dir))
        tissue_disp = max(0.0, tissue_disp)
        reward = self._reward(dist_grasp, tissue_disp)

        if self.phase == self.APPROACH and dist_grasp < self.grasp_radius:
            self.phase = self.GRASP
        if (self.phase == self.GRASP
                and self.jaw_angle < self.jaw_close_thr
                and dist_grasp < self.grasp_radius):
            self.grasped     = True
            self.grasp_point = self.ee_pos.copy()
            self.phase       = self.RETRACT
        # BUG FIX 1: threshold 0.8 -> 0.6
        if self.phase == self.RETRACT and tissue_disp >= 0.6 * self.target_height:
            self.phase = self.HOLD
        if self.phase == self.HOLD:
            self.hold_counter = self.hold_counter + 1 if tissue_disp >= 0.6 * self.target_height else 0

        # BUG FIX 1: hold_counter 10 -> 3
        self.success = self.hold_counter >= 3
        tear         = self.total_force > 2 * self.force_limit
        terminated   = self.success or tear
        truncated    = self.step_count >= self.max_steps

        # record trajectory for visualizer
        self._traj_ee.append(self.ee_pos.copy())
        self._traj_anchors.append(self.anchor_pos.copy())
        self._traj_jaw.append(self.jaw_angle)
        self._traj_phase.append(self.phase)
        self._traj_disp.append(tissue_disp)

        info = {'success': self.success, 'phase': self.phase,
                'tissue_disp': tissue_disp, 'dist_grasp': dist_grasp,
                'force': self.total_force, 'grasped': self.grasped}
        return self._obs(), reward, terminated, truncated, info

    def _deform_tissue(self):
        if not self.grasped:
            self.anchor_pos = self.anchor_pos * 0.95 + self.anchor_rest * 0.05
        else:
            disp = self.ee_pos - self.grasp_point
            for i in range(self.n_anchors):
                d = np.linalg.norm(self.anchor_rest[i] - self.grasp_target)
                falloff = float(np.exp(-d * 10.0))
                self.anchor_pos[i] = self.anchor_rest[i] + disp * falloff
        displacements    = self.anchor_pos - self.anchor_rest
        forces           = np.linalg.norm(displacements, axis=1) * self.spring_k
        self.total_force = float(forces.mean())

    def _reward(self, dist_grasp, tissue_disp):
        force_pen = -0.5 * float(np.log(1.0 + max(0.0, self.total_force - self.force_limit)))
        if self.phase == self.APPROACH:
            return float(-dist_grasp * 0.3 + force_pen)
        elif self.phase == self.GRASP:
            grasp_bonus = 10.0 if self.grasped else 0.0
            return float(-dist_grasp * 0.3 + grasp_bonus + force_pen)
        else:
            progress = tissue_disp / max(self.target_height, 1e-6)
            return float(progress * 0.4 + force_pen)

    def _obs(self):
        obs = np.concatenate([self.ee_pos, [self.jaw_angle],
                              self.joints, self.anchor_pos.flatten()]).astype(np.float32)
        return np.clip(obs, -5.0, 5.0)


def _test_env():
    for md in [False, True]:
        env = TissueRetractEnv(multidirectional=md, seed=0)
        obs, _ = env.reset()
        assert obs.shape == (23,)
        for _ in range(5):
            obs, r, te, tr, info = env.step(env.action_space.sample())
    print('env v2 sanity check passed')
