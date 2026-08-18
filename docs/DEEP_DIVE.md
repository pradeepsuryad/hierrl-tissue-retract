# HierRL-TissueRetract: Complete Deep Dive
## Every Cell, Every State, Every Design Choice — From Scratch

---

# PART A: THE REAL-WORLD PROBLEM

## What Is Tissue Retraction?

In many surgeries (liver resection, tumour removal, organ repair), the surgeon needs to **move a flap of tissue out of the way** to see and access what's underneath. This is called tissue retraction. It's one of the most common surgical sub-tasks.

A surgical robot (like da Vinci) must:

1. Move its gripper to the tissue flap
2. Close the gripper to grab the tissue
3. Pull the tissue to expose the surgical site
4. Hold it steady while the surgeon operates

This is fundamentally a **four-phase sequential task**. Each phase has completely different physics, goals, and failure modes. That's the core challenge.

## Why RL Instead of Classical Control?

You *could* hand-code a controller (the demo generator in Cell 12 does exactly this, achieving 96% success). Three reasons RL is still the right research direction:

**1. Privileged Information.** The scripted controller cheats — it reads `env.grasp_target` and `env.retract_dir` directly from the simulator's internal state. A real robot doesn't know the exact grasp point or pull direction. The RL agent must learn from observations only.

**2. Brittleness.** The scripted controller works for this exact spring-mass tissue model. Change the tissue stiffness, anchor layout, or retraction angle, and you'd need to rewrite the controller. RL generalises.

**3. Research Question.** The real question isn't "can we solve tissue retraction" — it's "can hierarchical RL with demonstrations solve multi-phase surgical tasks that flat RL cannot?" Your results prove it can: flat SAC gets 16.7%, ViSkill-DEX v3T gets 50% eval.

---

# PART B: THE MDP — EVERY DIMENSION EXPLAINED

An MDP (Markov Decision Process) is defined by five things: States (S), Actions (A), Transitions (T), Rewards (R), and Discount Factor (γ). Here is every detail.

## States: What the Agent Observes

### v1/v2: 23-dimensional observation vector

```
Index   Name            Size   What It Physically Means
─────   ──────────────  ────   ──────────────────────────────────────────────
[0:3]   ee_pos          3      End-effector (gripper tip) XYZ position in
                               world coordinates. The "hand" of the robot.
                               Range: clipped to [-1, 1] per axis.
                               Example: [0.05, -0.02, 0.18] means the
                               gripper is 5cm right, 2cm back, 18cm up.

[3]     jaw_angle       1      How open the gripper is.
                               0.0 = fully closed (clamping tissue)
                               1.0 = fully open (ready to grab)
                               0.15 = jaw_close_thr (the threshold
                               where the env considers the jaw "closed")

[4:11]  joints          7      Seven joint angles of the robot arm.
                               A 7-DOF (degree of freedom) robot like the
                               Kuka IIWA or da Vinci PSM has 7 joints:
                               
                               Joint 0 (shoulder rotate): left/right base rotation
                               Joint 1 (shoulder lift):   up/down at base
                               Joint 2 (elbow rotate):    twist of upper arm
                               Joint 3 (elbow flex):      bend at elbow
                               Joint 4 (wrist rotate):    twist of forearm
                               Joint 5 (wrist flex):      bend at wrist
                               Joint 6 (wrist roll):      roll of end-effector
                               
                               In THIS sim, joints are approximated:
                               joints[:3] += delta_pos * 0.5 (first 3 track
                               position changes loosely). Range: [-π, π].
                               
                               WHY include joints if we already have ee_pos?
                               Because in real robotics, the same ee_pos can
                               be reached by multiple joint configurations
                               (redundancy). The agent needs to know the arm
                               shape to predict what movements are possible
                               from here. Also, future transfer to real robots
                               (sim-to-real) requires joint-level observations.

[11:23] anchor_pos      12     4 tissue anchor points × 3 coordinates each.
                               These ARE the tissue deformation state.
                               
                               Anchor 0: [-0.05, -0.05, 0] + noise (bottom-left)
                               Anchor 1: [+0.05, -0.05, 0] + noise (bottom-right)
                               Anchor 2: [-0.05, +0.05, 0] + noise (top-left)
                               Anchor 3: [+0.05, +0.05, 0] + noise (top-right)
                               
                               They start near rest position (a flat square).
                               When the gripper grabs and pulls, anchors
                               displace based on spring physics. The agent
                               sees the CURRENT position of all 4 anchors,
                               so it knows exactly how deformed the tissue is.
```

### v3: 26-dimensional (adds goal-relative vector)

```
[23:26] ee_pos - grasp_target    3    The vector FROM the grasp target TO
                                      the current gripper position.
                                      
                                      WHY THIS FIXES THE EVAL GAP:
                                      In v2, the agent saw ee_pos = [0.12, 0.03, 0.18].
                                      During training, grasp_target was always near
                                      [0, 0, 0] ± 0.05 jitter. The agent memorized
                                      "move toward approximately [0, 0, 0]."
                                      
                                      At eval, jitter shifts the target to [0.04, -0.03, 0.02].
                                      The agent still tries to reach [0, 0, 0] and misses.
                                      
                                      The relative vector solves this:
                                      If ee = [0.12, 0.03, 0.18] and target = [0.04, -0.03, 0.02]:
                                      relative = [0.08, 0.06, 0.16]
                                      
                                      This tells the agent "you are 8cm right, 6cm forward,
                                      and 16cm above the target" — independent of where
                                      the target actually is. The agent learns to reduce
                                      this vector to zero, which works for ANY target position.
```

### v4: 29-dimensional (adds retraction direction)

```
[26:29] retract_dir     3      Unit vector indicating which direction to pull.
                               
                               In v1/v2: always [0, 0, 1] (straight up).
                               In multidirectional mode: random direction
                               within a cone (θ ∈ [0, π/3], φ ∈ [0, 2π]).
                               
                               Without this, the agent in multidirectional mode
                               has NO WAY to know which direction to pull.
                               It can only discover this through trial-and-error
                               each episode. Result: 0% eval success.
                               
                               With this, the agent knows "pull this way" and
                               can directly use the information.
```

## Actions: What the Agent Controls

```
Index   Name        Range     Physical Effect
─────   ──────────  ────────  ────────────────────────────────
[0:3]   delta_pos   [-1, 1]   Gripper movement per step.
                              Actual displacement = action × 0.05
                              So action=[1,0,0] moves 5cm in X.
                              action=[0,0,1] moves 5cm upward.
                              
                              WHY 0.05 SCALE?
                              Too large → jerky, overshoots the target.
                              Too small → takes forever to reach anything.
                              0.05 means the gripper needs ~4 steps to
                              cover 0.2m (the starting distance above target).

[3]     delta_jaw   [-1, 1]   Jaw open/close per step.
                              Actual change = action × 0.1
                              action[3] = -1 → close jaw by 0.1 per step
                              action[3] = +1 → open jaw by 0.1 per step
                              
                              Starting jaw_angle = 0.6.
                              Close threshold = 0.15.
                              So it takes at least (0.6-0.15)/0.1 = 4.5 → 5
                              steps of action[3]=-1 to close the jaw.
```

**Why continuous actions?** Robot motion is inherently continuous. Discretising "move gripper" into a handful of directions loses the fine control needed for precise grasping.

## Transitions: What Happens After an Action

Each `env.step(action)` does these things in order:

**1. Gripper moves:**
```python
ee_pos = clip(ee_pos + action[:3] * 0.05, -1, 1)
```

**2. Jaw opens/closes:**
```python
jaw_angle = clip(jaw_angle + action[3] * 0.1, 0, 1)
```

**3. Joint angles update** (approximate kinematic proxy):
```python
joints[:3] = clip(joints[:3] + action[:3] * 0.025, -π, π)
```
This is a simplified proxy — real robots use inverse kinematics. Here, the first 3 joints loosely track gripper movement.

**4. Tissue deforms** via spring-mass physics:
```python
if not grasped:
    # tissue relaxes toward rest position (spring return)
    anchor_pos = anchor_pos * 0.95 + anchor_rest * 0.05
else:
    # tissue follows gripper with distance-based falloff
    displacement = ee_pos - grasp_point
    for each anchor i:
        distance_from_grasp = ||anchor_rest[i] - grasp_target||
        falloff = exp(-distance × 10)  # close anchors move more
        anchor_pos[i] = anchor_rest[i] + displacement × falloff
```

**Why exponential falloff?** Real tissue doesn't move uniformly. The point you're pulling moves the most; tissue further away resists more. `exp(-d × 10)` creates this natural behavior.

**5. Forces computed:**
```python
displacements = anchor_pos - anchor_rest  # how far each anchor moved
forces = ||displacements|| × spring_k     # spring_k = 50
total_force = mean(forces)                 # average across anchors
```

**6. Phase transitions** (automatic, not agent-controlled):
```
APPROACH → GRASP:   when dist(ee_pos, grasp_target) < 0.06
GRASP → RETRACT:    when jaw_angle < 0.15 AND dist < 0.06
                    (this also sets grasped=True and records grasp_point)
RETRACT → HOLD:     when tissue_disp >= 0.6 × target_height
```

**7. Termination conditions:**
```
SUCCESS:    hold_counter >= 3 (tissue held for 3 consecutive steps)
TEAR:       total_force > 2 × force_limit = 10 (tissue ripped)
TRUNCATED:  step_count >= max_steps (ran out of time)
```

## Reward Function: What Drives Learning

```python
def _reward(dist_grasp, tissue_disp):
    # Force penalty: logarithmic, kicks in above force_limit=5
    force_pen = -0.5 × log(1 + max(0, total_force - 5))
    
    if phase == APPROACH:
        return -dist_grasp × 0.3 + force_pen
        # Pure distance-based: every step closer gets more reward
        # Typical: dist starts ~0.2, reward ~ -0.06
        
    elif phase == GRASP:
        grasp_bonus = 10.0 if grasped else 0.0
        return -dist_grasp × 0.3 + grasp_bonus + force_pen
        # Massive +10 spike when grasping succeeds
        # This is the critical sparse reward signal
        
    else:  # RETRACT or HOLD
        progress = tissue_disp / target_height
        return progress × 0.4 + force_pen
        # Reward proportional to how much tissue is pulled
        # At full retraction: progress ≈ 1, reward ≈ 0.4
```

**Why logarithmic force penalty?** Linear penalty would be too gentle at low forces and too harsh at high forces. Logarithmic is gentle at first (lets the agent explore) but grows sharply as force approaches tear threshold — like a soft warning that becomes urgent.

**Why +10 grasp bonus?** Grasping is the hardest transition. Without a large bonus, the agent might never discover that closing the jaw at the right moment is valuable — the natural reward signal is too small.

## Discount Factor: γ = 0.99

The agent cares about rewards ~100 steps into the future. This matters because the grasp bonus (+10) comes many steps after the approach decisions that made it possible. With γ = 0.9, the +10 bonus at step 20 would be worth only 10 × 0.9^20 = 1.2 at step 0. With γ = 0.99, it's worth 10 × 0.99^20 = 8.2 — much stronger learning signal.

---

# PART C: CELL-BY-CELL NOTEBOOK WALKTHROUGH

## Cell 0 — Title (Markdown)
```
# HierRL-TissueRetract — v2 (Patched)
```
Just the project title. "v2" means this is the second major version after bug fixes. "Patched" refers to the three bug fixes applied from v1.

## Cell 1 — Install Dependencies
```python
!pip install gymnasium torch numpy matplotlib scipy tqdm pandas -q
```
**Why each package:**
- `gymnasium`: The RL environment API. Provides `env.reset()`, `env.step()`, `observation_space`, `action_space`.
- `torch`: PyTorch neural network library. All actors, critics, and value networks are PyTorch modules.
- `numpy`: Array math for environment physics (spring-mass calculations).
- `matplotlib`: All plots — learning curves, 3D trajectories, phase timelines.
- `scipy`: Used for `scipy.stats` (smoothing).
- `tqdm`: Progress bars during training.
- `pandas`: CSV export of training logs.
- `-q`: Quiet mode — suppresses install output.

## Cell 2 — Imports
```python
import numpy as np
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
import copy, time
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec, matplotlib.patches as mpatches
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
```

**Why `copy`?** For `copy.deepcopy(critic)` — target networks in SAC/DDPG are deep copies of the main network.

**Why `deque(maxlen=30)`?** Used later for rolling averages of rewards and success rates. A deque with max length automatically discards old values — perfect for a sliding window.

**Why `device` detection?** All neural networks are placed on `device`. In Colab with GPU it uses CUDA; otherwise CPU. In practice everything ran on CPU because the networks are small (256×256 MLPs) and the bottleneck is the sequential environment stepping.

## Cell 3 — Markdown: Bug Fix Notice

## Cell 4 — TissueRetractEnv (THE ENVIRONMENT)

This is the core of the entire project. ~120 lines that define the complete MDP.

### `__init__`: Setting up the world
```python
self.n_anchors     = 4        # tissue modeled as 4 point masses
self.spring_k      = 50.0     # spring constant (N/m equivalent)
self.force_limit   = 5.0      # above this, force penalty kicks in
self.grasp_radius  = 0.06     # must be within 6cm to grab
self.jaw_close_thr = 0.15     # jaw_angle below this = "closed"
self.pos_scale     = 0.05     # action→position scaling
self.jaw_scale     = 0.1      # action→jaw scaling
```

**Why these specific numbers?** They're tuned so that a successful episode takes 15–40 steps (fast enough for 200k training steps to cover ~5000+ episodes) and failure modes are meaningful (tearing actually happens if you're rough, truncation if you're too slow).

### `reset()`: Starting a new episode
```python
self.grasp_target  = [0, 0, 0] + jitter(0.05)   # where to grab (±5cm noise)
self.target_height = 0.15 + uniform(0, 0.05)      # how far to pull (15-20cm)
self.ee_pos        = grasp_target + [0, 0, 0.2] + jitter(0.05)  # start 20cm above
self.jaw_angle     = 0.6                           # gripper starts half-open
```

**Why jitter?** Without it, every episode starts identically and the agent memorises one trajectory. Jitter forces the agent to generalise to different starting conditions.

### `_deform_tissue()`: Spring-mass physics
```python
if not grasped:
    anchor_pos = anchor_pos * 0.95 + anchor_rest * 0.05  # relax
else:
    for each anchor:
        falloff = exp(-distance_from_grasp × 10)
        anchor_pos[i] = anchor_rest[i] + gripper_displacement × falloff
```

The 0.95/0.05 blend is exponential smoothing — tissue slowly springs back when not held. The `exp(-d×10)` creates realistic deformation where close tissue follows the gripper and far tissue barely moves.

### `_test_env()`: Sanity check
Runs 5 random steps in both modes, asserts obs shape is (23,). If this fails, the notebook stops immediately — no wasted training on a broken environment.

## Cell 5 — Neural Network Architectures

### MLP (Multi-Layer Perceptron)
```python
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256,256)):
        # [Linear → LayerNorm → ReLU] × 2 → Linear
```

**Why (256,256)?** Standard size for continuous control. Smaller (64,64) would underfit the multi-phase task. Larger (512,512) would overfit with 200k steps.

**Why LayerNorm?** Normalises activations within each layer. Prevents the vanishing/exploding gradient problem that's especially bad with RL's non-stationary data. More stable than BatchNorm for RL because RL batches are small and non-IID.

### GaussianActor (for SAC)
```python
class GaussianActor(nn.Module):
    # Outputs: mean μ and log-std of a Gaussian distribution
    # Sampling: x ~ Normal(μ, exp(log_std))
    #           action = tanh(x)  ← squash to [-1, 1]
    #           log_prob adjusted for tanh squashing
```

**Why Gaussian + tanh squashing?** The action space is [-1, 1]. A raw Gaussian can output any value. `tanh` squashes to [-1, 1] naturally. The log-probability correction (`-log(1 - tanh²(x))`) accounts for the change of variables so that entropy calculations remain correct.

**Why clamp log_std to [-5, 2]?** 
- `log_std = -5` → std = 0.0067 (nearly deterministic)
- `log_std = 2` → std = 7.4 (very noisy)
Without clamping, the network could make std exactly 0 (division by zero) or infinity (useless noise).

### DeterministicActor (for DDPG)
```python
class DeterministicActor(nn.Module):
    # MLP with Tanh output activation → directly outputs action in [-1, 1]
```

No stochasticity. Exploration comes from OU noise added externally.

### DoubleQCritic
```python
class DoubleQCritic(nn.Module):
    # Two independent Q-networks: q1(s,a) and q2(s,a)
    # q_min(s,a) = min(q1, q2)
```

**Why two Q-networks?** This is the "clipped double-Q trick" from TD3. Q-learning overestimates values because it takes max over noisy estimates. Using the minimum of two independent estimates reduces overestimation bias.

### ValueNet
```python
class ValueNet(nn.Module):
    # Maps observation → single scalar V(s)
```
Used by ViSkill's meta-controller to estimate state value for skill selection.

## Cell 6 — ReplayBuffer

```python
class ReplayBuffer:
    # Pre-allocated numpy arrays for (obs, action, reward, next_obs, done)
    # Circular buffer: when full, overwrites oldest transitions
```

**Why pre-allocated?** Python lists with `.append()` are slow for 300k transitions. Pre-allocated numpy arrays with pointer arithmetic (`self.ptr = (self.ptr + 1) % capacity`) are O(1) per insert.

**Why 300k capacity?** At 200k training steps, with episodes ~30 steps long, you generate ~200k transitions. 300k ensures the buffer never needs to wrap around AND has room for demo transitions.

## Cell 7 — SAC (Soft Actor-Critic)

### Core SAC Loop

```
1. Sample batch from buffer
2. Compute target: y = r + γ(1-done) × [min(Q1_target, Q2_target) - α × log_prob]
3. Update critic: minimize MSE(Q1, y) + MSE(Q2, y)
4. Update actor: minimize α×log_prob - Q_min(s, new_action)
5. Update α: minimize -α × (log_prob + target_entropy)
6. Soft-update targets: θ_target = τ×θ + (1-τ)×θ_target
```

**Why entropy (the "soft" in SAC)?** The actor loss includes `α × log_prob`. This pushes the policy to be MORE random, not less. Why would you want randomness?

Because in RL, the agent needs to explore. Without entropy, SAC would converge to a deterministic policy that exploits one trajectory. With entropy, it maintains stochasticity — exploring different grasping angles, approach paths, etc. The temperature α is automatically tuned so entropy stays at `target_entropy = -act_dim = -4`.

**Why `warmup_steps = 1000`?** The agent collects 1000 random transitions before any learning. This fills the replay buffer with diverse experiences. Training on an empty or tiny buffer produces unstable gradients.

**Why `tau = 0.005`?** Polyak averaging for target network updates. Instead of copying weights every N steps (hard update), we blend: `target = 0.005 × main + 0.995 × target`. This smooths the target, preventing oscillation.

**Why gradient clipping to 1.0?** RL loss landscapes are volatile. A single bad batch can produce enormous gradients that destroy learned weights. Clipping prevents this.

## Cell 8 — DDPG (Deep Deterministic Policy Gradient)

### How DDPG Differs from SAC
- **Actor**: Deterministic (outputs one action, not a distribution)
- **Exploration**: Ornstein-Uhlenbeck noise (correlated noise, not white noise)
- **No entropy term**: No automatic exploration pressure
- **Single critic update**: Same double-Q trick as SAC

### OU Noise
```python
x += -θ×x×dt + σ×√dt×N(0,1)
```
**Why OU instead of white noise?** OU noise is temporally correlated — if you pushed right last step, you're more likely to push right this step too. This creates coherent exploration trajectories rather than random jitter, which is important for reaching distant goals.

## Cell 9 — DEX (Demo-augmented Exploration)

### How DEX Extends SAC
DEX wraps SAC and adds a behavioral cloning (BC) loss:

```python
actor_loss = SAC_loss + λ_bc × MSE(policy_output, demo_action)
```

**λ_bc annealing:** Starts at 1.0 (100% imitation), linearly decays to 0.0 over `anneal_steps`. The idea: start by imitating demos, gradually transition to pure RL.

**The annealing collapse problem:** DEX peaked at 50% success then crashed to 10%. Why? When λ_bc reaches 0, the RL component isn't strong enough to sustain the learned behavior. The policy "forgets" the demo behavior before discovering a good RL policy. This is the central diagnostic finding of the paper.

### Demo buffer management
```python
def add_demo(self, obs, action, reward, next_obs, done):
    idx = self.sac.buffer.ptr
    self.sac.buffer.add(...)
    self.demo_indices.append(idx)
```
Demos go into the SAME buffer as RL transitions. The BC loss only samples from `demo_indices`. This means demos are never overwritten (they were added early when `ptr` was low) and the agent always has access to them.

## Cell 10 — ViSkill (Hierarchical Framework)

### Architecture
```
ViSkill
├── skills[0]: SAC for APPROACH
├── skills[1]: SAC for GRASP  
├── skills[2]: SAC for RETRACT
├── skills[3]: SAC for HOLD
└── value_net: ValueNet for meta-controller
```

Each sub-agent is a complete SAC instance with its own actor, critic, target networks, and replay buffer.

### Skill Selection
```python
def select_action(self, obs, env_phase=None, deterministic=False):
    if deterministic and self._meta_reliable:
        skill = self._meta_select(obs)      # learned selector
    elif env_phase is not None:
        skill = min(int(env_phase), 3)       # use env phase
    else:
        skill = self.current_skill           # continue current
```

During training: the environment phase directly selects which sub-agent acts. This is called "oracle phase labeling" — the agent doesn't need to figure out which phase it's in.

During eval (after 5000 steps): the value network predicts V(s) and uses percentile thresholds to select the skill. Low V → early phase (APPROACH), high V → late phase (HOLD).

### Sub-Agent Rewards (Different from Global Reward!)
```python
APPROACH:  -dist × 0.5 + force_penalty        (get closer)
GRASP:     +5.0 if grasped else -dist × 0.3   (close jaw)
RETRACT:   tissue_disp / 0.15 × 0.5           (pull up)
HOLD:      +1.0 if disp > 0.09 else -0.2      (stay steady)
```

**Why different from global reward?** The global reward is sparse — the agent only gets meaningful signal when it transitions between phases. Sub-agent rewards are DENSE — every step has informative reward for the current sub-task. Dense rewards make learning dramatically faster.

### Jaw Bias Fix (Bug Fix 3)
```python
if skill == 1 and not deterministic:
    action[3] = clip(action[3] - 0.5, -1, 1)
```
During training, the GRASP sub-agent's action[3] is biased toward closing. Without this, SAC's entropy bonus pushes the jaw open and closed equally, making grasping nearly impossible.

### Store Function
```python
def store(self, obs, action, reward, next_obs, done, env_phase, info):
    skill_id = min(int(env_phase), 3)
    sub_reward = self._skill_reward(info, skill_id)
    self.skills[skill_id].buffer.add(obs, action, sub_reward, next_obs, done)
    self.skills[0].buffer.add(obs, action, reward, next_obs, done)
```

**Two buffer writes:** The transition goes into (1) the active sub-agent's buffer with sub-reward, and (2) skill[0]'s buffer with global reward. Skill[0]'s buffer is shared across all skills for the value network training.

## Cell 11 — Markdown: Demo Generator Section

## Cell 12 — Demo Generator (Scripted Controller)

```python
def generate_demos(env, n_episodes=150, noise_scale=0.04):
```

This is the expert that creates demonstration data. It uses privileged information (env.grasp_target, env.retract_dir, env.phase).

### Phase-by-phase controller logic:

**APPROACH:** 
```python
delta = grasp_target - ee_pos   # vector toward target
delta = delta / ||delta||        # normalize to unit vector
action = [delta × 0.9, 0.3]     # move fast, keep jaw slightly open
```

**GRASP (Bug Fix 2 — the critical fix):**
```python
if jaw_angle > jaw_close_thr + 0.05:
    action = [0, 0, 0, -1]      # STOP MOVING. Slam jaw shut.
else:
    action = [0, 0, 0.01, -1]   # Tiny upward nudge + keep closing
```

The original v1 mixed jaw closing with movement. The problem: the env checks `jaw < 0.15 AND dist < 0.06` simultaneously. If the gripper is moving while closing the jaw, it might drift outside the 0.06 radius before the jaw finishes closing → grasping NEVER triggers. Fix: freeze position, close jaw first.

**RETRACT/HOLD:**
```python
action = [retract_dir × [0.3, 0.3, 1.0], 0.0]
# Move in retraction direction, jaw stays clamped
```

The Z component is scaled to 1.0 (full speed upward) while X/Y are 0.3 (slower lateral movement). This prioritizes upward retraction.

**Noise:** `action += N(0, 0.04)` — small Gaussian noise for generalization. Too much noise → bad demos. Too little → agent memorizes exact trajectories.

**Result:** 96% scripted success, ~5400 transitions from 150 episodes.

## Cell 13 — Train Function

```python
def train(name, agent, env, total_steps=200000, log_every=5000):
```

The universal training loop for ALL algorithms:

```
for each step:
    1. Get action from agent (with exploration)
    2. Step environment
    3. Store transition in agent's buffer
    4. Call agent.update() (learn from buffer)
    5. If episode done: reset, log stats
    6. Every log_every steps: record success rate, reward
```

**ViSkill special handling:** ViSkill's `select_action` takes `env_phase`, and `store` takes `env_phase` and `info` for skill-specific reward computation.

**DEX special handling:** DEX transitions go to `agent.sac.buffer` (not `agent.buffer` directly).

## Cell 14 — Configuration

```python
TOTAL_STEPS     = 200000    # why 200k? SAC/DDPG converge here
LOG_EVERY       = 5000      # log every 5k steps ≈ 40 data points
N_DEMO_EPISODES = 150       # enough demos for 96% coverage
OBS_DIM, ACT_DIM = 23, 4   # matching TissueRetractEnv
```

## Cells 16–19 — Training All Four Algorithms

### Cell 16: SAC Training
Creates `SAC(23, 4)` and trains 200k steps. No demos. Pure RL from scratch.
**Result: 16.7% final success.** Reward eventually positive (~27) but success rate stays low. SAC learns to approach and sometimes grasp, but rarely completes retraction+hold.

### Cell 17: DDPG Training
Creates `DDPG(23, 4)` and trains 200k steps. No demos.
**Result: 23.3% final success.** Better than SAC despite lower reward (-27). DDPG's deterministic policy is better at the precise jaw-closing action.

### Cell 18: DEX Training
Generates 150 demos first, loads them into buffer, then trains 200k steps.
**Result: 10% final (50% peak).** The annealing collapse — performance peaks early then crashes when BC weight reaches 0.

### Cell 19: ViSkill Training
Creates 4 SAC sub-agents. No demos for vanilla ViSkill.
**Result: 60% train, 10% eval.** Massive improvement over flat methods, but huge train-eval gap.

## Cells 20–22 — Animation Visualizer

Runs eval episodes for each algorithm, records trajectories (ee_pos, anchor_pos, jaw, phase, displacement), and renders 3D animations with:
- Left panel: 3D gripper trajectory colored by phase
- Right panels: jaw angle and tissue displacement over time

**Why visualize?** Numbers alone don't tell you WHY an algorithm fails. The animation shows SAC's gripper oscillating near the target without grasping, DDPG's efficient but rough approach, DEX's clean trajectory that falls apart, and ViSkill's phase-by-phase execution.

## Cells 23–24 — Static 3D Trajectory Comparison

Four side-by-side 3D plots showing all four algorithms' gripper paths. Dashed lines = initial tissue shape, solid lines = final tissue shape. Phase colors show where each algorithm spends its time.

## Cells 25–26 — Training Results Dashboard

Four-panel figure:
1. Success rate over training (smoothed)
2. Episode reward over training (smoothed)
3. Final success rate bar chart
4. Summary statistics table

## Cell 27 — ViSkill Skill Timeline

Shows which sub-agent was active at each timestep across 5 eval episodes. Clean phase transitions (purple→green→orange→brown) indicate the hierarchical decomposition is working.

## Cell 28 — CSV Export

Saves all training logs to `training_logs_v2.csv` for offline analysis.

## Cells 30–32 — V3 Extension (ViSkill-DEX)

### Cell 30: TissueRetractEnvV3
Extends the environment with `ee_pos - grasp_target` → 26D observation.

### Cell 31: Phase-Specific Demo Generator
Instead of one big demo list, generates separate demo lists per phase:
```python
phase_demos = {0: [...], 1: [...], 2: [...], 3: [...]}
```
Each sub-DEX agent gets ONLY demos from its own phase.

### Cell 32: ViSkillDEXTuned (SubDEXTuned)
The main v3T contribution. Key changes:

**BC Floor:** `lambda_min = 0.15` for HOLD. λ_bc never drops below 0.15, preventing the annealing collapse that killed DEX.

**Extended annealing:** 100k steps instead of 40k. Slower decay → smoother transition from imitation to RL.

**Lower warmup for HOLD:** 50 steps instead of 300. HOLD has few transitions (tissue is barely moving), so waiting 300 steps meant HOLD never started learning. 50 lets it begin immediately.

**Continuous HOLD reward:** Changed from binary (+1 / -0.2) to continuous based on displacement. This gives gradient signal for "almost holding" rather than cliff-edge success/failure.

**Result: 56.7% train, 50% eval. Gap = 6.7pp.** Best configuration.

## Cells 33–34 — V3T Eval Visualization

Runs individual eval episodes with animation to verify the tuned agent works.

## Cells 35–36 — Zero-Shot Multidirectional Test

Tests v3T (trained on upward-only) on random retraction angles. Shows that without direction information, the agent can't generalise to new pull directions.

## Cells 37–39 — Multidirectional Training

Generates 300 demos with random retraction angles, trains a new ViSkillDEXTuned agent for 300k steps on multidirectional env (26D).

**Result: 0% eval on seeds 200-209.** Even with multidirectional training, the 26D observation doesn't include the retraction direction — the agent can't solve the task.

## Cells 40–43 — Multidirectional Evaluation

Batch evaluation on 10 seeds confirming 0% success. The failure analysis shows most episodes fail at RETRACT — the agent doesn't know which direction to pull.

## Cells 44–45 — V4 (29D observation)

Adds `retract_dir` (3D vector) to observation → 29D. Generates demos and initializes training. **Not yet completed** — marked as optional for submission.

## Cells 46–47 — V4 Evaluation (Not Run)

Would evaluate the V4 agent if training completed.

---

# PART D: SUMMARY TABLE

```
Version   Obs   Sub-agents   Key Change                Train%   Eval%   Gap
──────    ───   ──────────   ─────────────────────     ──────   ─────   ────
SAC       23    flat (1)     baseline                   16.7%    —       —
DDPG      23    flat (1)     deterministic policy       23.3%    —       —
DEX       23    flat (1)     +demos                     10.0%    —       —
ViSkill   23    4 × SAC      hierarchical               60.0%   10.0%   50pp
v3        26    4 × DEX      +per-phase demos, +goal    80.0%   30.0%   50pp
v3T       26    4 × DEX      +BC floor, +tuned HOLD     56.7%   50.0%   6.7pp
v3 multi  26    4 × DEX      multidirectional train      0.0%    0.0%    —
v4        29    4 × DEX      +retract_dir               (not run)
```
