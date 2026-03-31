# Long-Horizon Simulation Benchmarks for Representation Learning Evaluation

*Exported conversation — 2026-03-21*

---

## Context

I am doing representation learning with videos, so that these representations lend to better planning. I will do this pretraining in stages:

- **Stage 1:** Linear recurrence over short videos, so the model learns to compress videos into one spatio-temporal state with dynamics. I use future prediction of DINO features for this.
- **Stage 2:** Future prediction over long horizons across chunks.

I am trying to find how to evaluate such representations. Ideally, I could do a long horizon sim, where you have to make coffee — there is planning across multiple task stages etc. But I need to find such sims.

### Observations on Existing Benchmarks

- **DMC** — requires only 3 frames context (per [CroBo, arXiv:2603.13904](https://arxiv.org/abs/2603.13904))
- **Franka** — uses only current frames
- **NE-Dreamer** ([arXiv:2603.02765](https://arxiv.org/abs/2603.02765)) — uses rooms in DMC which seems slightly longer horizon
- Looking for something like [this paper](https://openreview.net/pdf?id=79BOATBal9) — truly long-horizon multi-stage tasks

### Referenced Papers

- **CroBo** — *Pixel-level Scene Understanding in One Token: Visual States Need What-is-Where Composition* ([arXiv:2603.13904](https://arxiv.org/abs/2603.13904)): Visual state representation learning via global-to-local reconstruction with a bottleneck token. Evaluated on vision-based robot policy learning benchmarks.
- **NE-Dreamer** — *Next Embedding Prediction Makes World Models Stronger* ([arXiv:2603.02765](https://arxiv.org/abs/2603.02765)): Decoder-free MBRL agent using a temporal transformer to predict next-step encoder embeddings. Tested on DMC and DMLab tasks involving memory and spatial reasoning.

---

## 🏆 Tier 1: Truly Long-Horizon, Multi-Stage Tasks

### 1. CALVIN (Composing Actions from Language and Vision)

- **What:** Tabletop manipulation with a Franka Panda arm; **34 distinct skills** composed into chains of up to **5 sequential instructions**
- **Why it fits:** Tasks are *compositional* — the agent must chain skills like "open drawer → place block → close drawer → push slider → press button." This is exactly the kind of multi-stage sequential planning that Stage 2 long-horizon chunk-based prediction should help with.
- **Horizon:** ~hundreds of environment steps per chain
- **Data:** 24 hours / 2.4M interaction steps of teleoperated "play" data across 4 environments
- **Details:**
  - 4 simulated tabletop environments (A, B, C, D) with a 7-DOF Franka Emika Panda arm
  - Variations in surface textures, object placements, lighting, and distractors
  - Multimodal sensing: RGB-D from static + gripper cameras, proprioception, tactile
  - Supports zero-shot evaluation on novel language instructions and unseen environments
  - Won 2022 IEEE RA-L Best Paper Award
- **Link:** [github.com/mees/calvin](https://github.com/mees/calvin)

### 2. LIBERO (Lifelong Learning Benchmark)

- **What:** Includes a **LIBERO-Long** suite with 10 explicitly long-horizon, multi-step manipulation tasks
- **Why it fits:** Tasks require compositional multi-step instructions; tests knowledge transfer across sequential tasks. Good for probing whether learned representations generalize across task stages.
- **Horizon:** Long compositional sequences
- **Details:**
  - Focuses on lifelong learning in decision-making (LLDM)
  - Transfer of declarative knowledge (object concepts, spatial info) and procedural knowledge (actions, behaviors)
  - Extendable procedural generation pipeline for diverse language-conditioned manipulation tasks
  - Research finding: simple sequential fine-tuning can outperform complex lifelong learning methods
- **Link:** [github.com/Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

### 3. BEHAVIOR-1K (via OmniGibson)

- **What:** **1,000 everyday household activities** (derived from human surveys) including things like making coffee, setting a table, doing laundry — each requiring **hundreds to thousands of environment steps**
- **Why it fits:** This is the closest to the "make coffee" style long-horizon eval. Truly long-horizon household tasks with multiple stages, state changes (temperature, wetness, toggled), and complex manipulation. The "make coffee" task literally exists here.
- **Horizon:** Hundreds to thousands of steps
- **Details:**
  - Sim: OmniGibson (Nvidia Omniverse + PhysX 5)
  - 50 fully interactive scenes, 9,000+ object models with detailed physical and semantic annotations
  - Extended object states: temperature, toggled status, soaking, dirtiness
  - Activities defined with initial and goal conditions in predicate logic
  - Evolution from iGibson 2.0
- **Link:** [behavior.stanford.edu](https://behavior.stanford.edu/)

### 4. VirtualHome

- **What:** 3D household sim where agents execute multi-step activity programs (e.g., "make coffee with milk" = open cabinet → grab cup → walk to coffee machine → …)
- **Why it fits:** Tasks are decomposed into action programs from natural language. Very long-horizon with clear sub-goal structure. Great for evaluating whether representations capture the abstract task structure needed for planning.
- **Details:**
  - Developed by MIT and University of Toronto
  - Uses natural language to create a database of chores
  - Translates verbal descriptions into simple code to drive virtual agents
  - Tasks range from simple actions to complex sequences like "make coffee with milk"
- **Link:** [github.com/xavierpuigf/virtualhome](https://github.com/xavierpuigf/virtualhome)

---

## 🥈 Tier 2: Moderate Horizon with Planning Requirements

### 5. RLBench

- **What:** 100 manipulation tasks ranging from simple (reach, push) to complex multi-step (empty dishwasher, place tray in oven)
- **Why it's useful:** The harder tasks (e.g., "empty dishwasher" = open door → slide tray → grasp plate → lift) are genuinely multi-stage. Good middle ground.
- **Details:**
  - Rich visual observations: RGB, depth, segmentation masks from multiple cameras
  - Sparse rewards — success defined at task completion
  - Supports few-shot learning and infinite demonstration generation
- **Link:** [github.com/stepjam/RLBench](https://github.com/stepjam/RLBench)

### 6. DMLab (DeepMind Lab) — Memory/Navigation Subset

- **What:** First-person 3D navigation with memory and spatial reasoning requirements
- **Why:** As noted from the NE-Dreamer paper, the memory-demanding subset of DMLab tasks (rooms, goal navigation) requires maintaining representations over longer horizons. Not manipulation, but tests temporal memory well.

### 7. Habitat 2.0 / 3.0 (Home Assistant Benchmark)

- **What:** Photorealistic 3D home environments with interactive tasks (tidy house, prepare groceries, set table) requiring navigation + manipulation
- **Why:** Combines navigation planning with manipulation — truly long horizon since the agent must move through rooms, find objects, and execute multi-step tasks. Habitat 3.0 adds human-robot interaction with humanoid simulation.
- **Link:** [aihabitat.org](https://aihabitat.org/)

---

## 🥉 Tier 3: Shorter Horizon (for reference/contrast)

### 8. Meta-World (ML45)

- Diverse manipulation tasks but generally **single-stage, short-horizon**
- Useful for meta-RL and multi-task learning, but won't stress-test long-horizon planning

### 9. DMC (DeepMind Control Suite)

- Most tasks need only ~3 frames of context
- Good for Stage 1 verification but won't stress-test Stage 2

---

## Other Benchmarks Mentioned in Search

| Benchmark | Focus | Notes |
|-----------|-------|-------|
| **RoboCerebra** | High-level reasoning in long-horizon manipulation | Hierarchical: VLM planner + VLA controller |
| **RoboMME** | Memory in VLA models | 16 tasks, hundreds to 1000+ steps |
| **RoboCAS** | Complex object arrangement | Tests spatial reasoning and long-horizon planning |
| **MImE** | Multi-task imitation (PyBullet) | UR5-Breakfast environment |
| **ThreeDWorld (TDW)** | Physical simulation (Unity) | Transport Challenge for multi-step manipulation |

---

## Recommendation Summary

| Stage | What to evaluate | Best benchmarks |
|-------|-----------------|-----------------|
| **Stage 1** (short video, linear recurrence → compress to spatio-temporal state) | Does the state capture dynamics? | **DMC**, **RLBench (simple tasks)**, probing tasks (future DINO feature prediction accuracy, perceptual straightness) |
| **Stage 2** (long-horizon chunk prediction) | Does the representation support planning across task stages? | **CALVIN** (5-chain tasks), **BEHAVIOR-1K** (make coffee!), **LIBERO-Long**, **VirtualHome** |

### Top Pick

If picking one for the "make coffee" style long-horizon eval: **BEHAVIOR-1K / OmniGibson** is exactly that.

**CALVIN** is a strong second choice because it's more established in the representation learning community and easier to set up, though the tasks are less complex than real household activities.
