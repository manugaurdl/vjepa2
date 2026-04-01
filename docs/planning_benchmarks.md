# Simulation Benchmarks for Action-Conditioned Recurrent World Model

## Context

The current project explores **recurrence as persistent memory** for long-video understanding, generation, and planning. A recurrent encoder built on frozen DINOv2 frame features uses predictive coding to maintain and update a latent state:

```
Error   = f(x_t) - W_pred(S_{t-1})
S_t     = S_{t-1} + W(Error)
Loss    = CE + lambda * L2(Error)
```

This has been validated on Something-Something V2 (action classification, 174 classes), where the recurrent encoder outperforms concat-frames / transformer baselines.

The next step is to make the world model **action-conditioned** and use it for **planning**: given the current latent state, roll out the model under different action sequences and pick the one that best reaches a goal. This document lays out a progressive set of simulation benchmarks for validating this capability.

---

## Why these benchmarks

The model's key properties -- recurrent state, predictive coding, DINO-feature space -- are most useful in **partially observable** environments where:

- **Memory matters**: the agent cannot see the full state at once (hidden objects, rooms already visited)
- **Dynamics are consistent**: next-observation prediction is meaningful and learnable
- **Planning requires lookahead**: naive reactive policies fail; the agent must simulate future states

All benchmarks below use discrete actions, support visual observations, and have increasing planning-horizon requirements.

---

## Architectural change: action conditioning

The prediction head currently predicts the next latent from state alone:

```
z_pred = W_pred(S_{t-1})
```

For planning, it becomes action-conditioned:

```
z_pred = W_pred(S_{t-1}, a_t)
```

where `a_t` is a one-hot (or learned-embedding) representation of the discrete action. The simplest implementation concatenates the action embedding to the state before the MLP:

```python
action_emb = self.action_embed(a_t)          # (B, D_action)
combined   = torch.cat([state, action_emb], dim=-1)  # (B, D + D_action)
z_pred     = self.predictor(combined)          # (B, D)
```

The rest of the predictive-coding update stays the same. The error signal `L2(Error)` now teaches the model how actions affect the environment.

---

## Tier 1: Fast iteration -- validate the mechanism

### 1. MiniGrid / BabyAI

| Property | Value |
|---|---|
| Actions | 7 discrete (left, right, forward, pickup, drop, toggle, done) |
| Observations | 7x7 partial grid (agent sees a cone ahead) |
| Speed | ~1000+ env steps/sec on CPU |
| Install | `pip install minigrid` |
| Key baselines | MuZero, Dreamer, IRIS |

**Progressive tasks (short to long horizon):**

| Task | Horizon | What it tests |
|---|---|---|
| `MiniGrid-Empty-8x8-v0` | ~10 steps | Sanity check: can the world model predict trivial navigation? |
| `MiniGrid-DoorKey-8x8-v0` | ~20-30 steps | First real planning task: find key, pick up, unlock door, reach goal |
| `MiniGrid-MultiRoom-N4-S5-v0` | ~50-80 steps | Memory of visited rooms under partial observability |
| `BabyAI-BossLevel-v0` | ~100+ steps | Compositional language instructions, multiple subtasks |

**Limitation**: visually trivial (simple colored tiles). DINO features won't be stressed here. That's fine -- the goal at this tier is to validate that action-conditioned prediction + tree search works at all.

**Milestone**: solve `DoorKey-8x8` with the learned world model + planning. This is the simplest task requiring genuine multi-step reasoning (find key -> pick up -> navigate to door -> unlock -> reach goal).

### 2. Crafter

| Property | Value |
|---|---|
| Actions | 17 discrete |
| Observations | 64x64 RGB (top-down 2D) |
| Achievements | 22, ranging from trivial to ~15 sequential subtasks |
| Install | `pip install crafter` |
| Key baselines | DreamerV3, DreamerV2, IRIS, PPO |

A 2D Minecraft-like survival game. Planning horizons range from ~5 steps (basic achievements like "collect wood") to 500+ steps (hard achievements like "make diamond pickaxe" requiring a chain of ~15 prerequisite subtasks). This is a **standard world-model benchmark** used in Dreamer papers.

**What it tests**: diverse skills, long-horizon planning, exploration, resource management. Richer visual dynamics than MiniGrid.

---

## Tier 2: Richer visuals, realistic dynamics

### 3. VizDoom

| Property | Value |
|---|---|
| Actions | Discrete (varies per scenario, typically 3-8) |
| Observations | First-person RGB (configurable resolution) |
| Install | `pip install vizdoom` |
| Key baselines | Dreamer, Plan2Explore, LEXA |

**Progressive tasks:**

| Task | Horizon | What it tests |
|---|---|---|
| `VizdoomBasic` | ~10 steps | Simple targeting |
| `VizdoomMyWayHome` | ~50-100 steps | Maze navigation, spatial memory |
| `VizdoomHealthGathering` | ~200+ steps | Survival, longer planning horizon |

First-person 3D environments with stylized but spatially meaningful visuals. DINO features become more informative here than in grid worlds -- they encode spatial structure, depth cues, and object semantics even in stylized scenes.

---

## Tier 3: Photorealistic -- DINO features shine

### 4. Habitat (PointNav / ObjectNav)

| Property | Value |
|---|---|
| Actions | 4 discrete (forward 0.25m, turn left 30deg, turn right 30deg, stop) |
| Observations | Photorealistic first-person RGB (from Matterport3D / HM3D scans) |
| Install | `pip install habitat-sim habitat-lab` + HM3D scene datasets |
| Key baselines | DINO-WM, SemExp, OVRL, Habitat challenge entries |

This is where the DINO-feature-based world model has the strongest advantage. DINO encodes rich semantic and geometric information from realistic indoor scenes. **DINO-WM already establishes baselines here**, making direct comparison straightforward.

**Progressive tasks:**

| Task | Horizon | What it tests |
|---|---|---|
| **PointNav** | ~50-200 steps | Navigate to (x,y) coordinates. Spatial memory + planning. |
| **ObjectNav** | ~100-500 steps | Find an object category ("find a bed"). Semantic understanding + exploration + memory. |

### 5. AI2-THOR / ALFRED (stretch goal)

| Property | Value |
|---|---|
| Actions | Discrete: navigation + object interaction (pick up, put, open, close, slice, toggle) |
| Observations | Photorealistic first-person RGB |
| Key baselines | FILM, HLSM, LLM-Planner |

Interactive household environments. ALFRED adds language-conditioned long-horizon tasks ("put a hot potato on the counter" requires: find potato -> pick up -> go to microwave -> heat -> go to counter -> put down). Tests the full stack: memory, dynamics prediction, multi-step planning with subgoals.

---

## Recommended progression

```
MiniGrid DoorKey ──validate mechanism──> MiniGrid MultiRoom
       │
       └──scale horizon──> Crafter
                              │
                              └──richer visuals──> VizDoom MyWayHome
                                                       │
                                                       └──photorealistic──> Habitat PointNav
                                                                                │
                                                                                └──semantic planning──> Habitat ObjectNav
```

---

## Planning algorithms

With discrete actions, the natural planning algorithms (in order of complexity) are:

### 1. Exhaustive tree search (start here)
- Enumerate all action sequences up to depth `d`
- Evaluate each using world-model rollouts; pick the sequence whose predicted final state is closest to the goal
- Feasible for MiniGrid: 7 actions, depth 5 = 16,807 rollouts (fast with a learned latent model)

### 2. MCTS (Monte Carlo Tree Search)
- Like MuZero: use the world model as the simulator inside MCTS
- Balances exploration vs exploitation via UCB
- Scales to longer horizons (Crafter, VizDoom)

### 3. CEM (Cross-Entropy Method) over action sequences
- Sample N random action sequences of length H
- Roll out each through the world model
- Keep top-K, refit a categorical distribution, resample
- Simple and parallelizable on GPU

---

## Pipeline for each benchmark

```
1. COLLECT TRAJECTORIES
   - Random policy or pretrained policy
   - Store: (obs_t, action_t, obs_{t+1}, reward_t, done_t)

2. TRAIN ACTION-CONDITIONED WORLD MODEL
   - Encode obs_t with frozen DINOv2 -> frame features
   - Recurrent encoder updates state: S_t = f(S_{t-1}, obs_t)
   - Predict next latent: z_pred = W_pred(S_t, a_t)
   - Loss = L2(z_pred - encode(obs_{t+1}))  [+ optional reward prediction]

3. PLAN AT TEST TIME
   - Encode current observation history into state S_t
   - For candidate action sequences, roll out world model in latent space
   - Score each rollout (distance to goal state, predicted reward, etc.)
   - Execute best first action, re-plan at next step (MPC-style)
```

---

## Implementation roadmap

| Step | Task | Key deliverable |
|---|---|---|
| 1 | Set up MiniGrid/BabyAI, collect trajectory dataset | `collect_trajectories.py`, trajectory HDF5/pickle files |
| 2 | Extend recurrent encoder with action conditioning | Modified `rnn.py` with `W_pred(S, a)` |
| 3 | Train world model on MiniGrid trajectories | Training script, loss curves, prediction visualizations |
| 4 | Implement tree search / MCTS planning | `planner.py` with tree search and MCTS |
| 5 | Evaluate on MiniGrid-DoorKey | Success rate, planning horizon, comparison to baselines |
| 6 | Scale to Crafter | Achievement scores, comparison to DreamerV3 |
| 7 | Move to Habitat PointNav/ObjectNav | SPL, success rate, comparison to DINO-WM |

---

## Key references

| Paper | Relevance |
|---|---|
| Dreamer (Hafner et al., 2020) | Latent world model + imagination-based planning; benchmarked on DMC, Atari |
| DreamerV3 (Hafner et al., 2023) | Scales Dreamer to diverse domains including Crafter |
| TD-MPC2 (Hansen et al., 2024) | Model-based RL with latent dynamics + CEM planning |
| DINO-WM (Zhou et al., 2024) | Uses DINOv2 features for world modeling; benchmarked on Habitat |
| MuZero (Schrittwieser et al., 2020) | MCTS with learned world model |
| IRIS (Micheli et al., 2023) | Transformer world model; tested on Atari and MiniGrid |
