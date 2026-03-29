# Stage 1: Recurrent Video Compression

## Goal
Train a recurrent encoder that compresses short video chunks into a single compact spatio-temporal state. This state must be expressive enough that stage 2 (long-horizon forecasting over compressed states) performs nearly as well as forecasting over uncompressed features.

## Architecture
- Frozen DINO backbone extracts per-frame features
- Recurrent encoder (TTT/GRU/etc.) compresses a sequence of DINO features into a single state S
- Decoder cross-attends to S and predicts future DINO features (passed through a linear layer)
- State dimensionality: TBD (start with single state vector, sweep later)
- DINO features: cls tokens vs patches undecided (start with one, ablate)

## Training
- **Primary loss**: MSE on predicted future DINO features (start simple, consider contrastive later if mode collapse appears)
- **Auxiliary loss**: Scene flow (self-supervised, encourages encoding motion/dynamics not just appearance)
- Frozen DINO backbone throughout stage 1 (cheap iteration, easy dataset swapping)

## Datasets (by length)

| Dataset | Avg Length | Why Use It |
|---|---|---|
| SSv2 | 2-6s | Temporal reasoning required. Fast ablation loop. Current dataset. |
| Kinetics-700 | ~10s | Scene/object diversity. Tests if state generalizes beyond SSv2's constrained scenes. |
| Diving48 | 3-5s | Fine-grained temporal discrimination (dive types differ only in motion sequence). |
| ActivityNet | ~1min | Multi-stage activities. Tests if state captures temporal structure over longer clips. |
| COIN | ~1min | Instructional with step annotations. Maps to compositional task stages. |
| Epic-Kitchens-100 | 10-20min | Egocentric, multi-step cooking. Closest proxy to stage 2 demands without Ego4D scale. |

**Ablation order**: SSv2 first, then Kinetics-700 for diversity, then Epic-Kitchens-100 for length.

## Proxy Evaluations

### Primary metric: Multi-horizon prediction decay curve (Eval A)
Freeze the encoder. Train a small MLP probe to predict the **raw DINO features** (i.e., frozen backbone output, before compression) of future frames/chunks at t+k, given only the compressed state S_t. Measure MSE between predicted and actual DINO features for k = 1, 2, 4, 8. Plot prediction quality vs. horizon. A flat curve = dynamics info is retained in S. A steep drop = compression is losing information needed for forecasting. The target is always uncompressed DINO features, not compressed states (that would be circular).

**This is the single most important eval because it directly measures what stage 2 needs.**

### Secondary metric: SSv2 linear probe (Eval E)
Freeze the state. Train a linear classifier for activity recognition on SSv2. SSv2 requires temporal reasoning ("pushing left to right" vs "right to left"). A high accuracy means dynamics are linearly accessible in the state.
baseline: concatenated frames DINO feats

### Diagnostic: Information lost during compression
Compare: (a) predictor with full DINO features of chunk t predicting chunk t+1, vs (b) same predictor with only compressed state of chunk t. The gap = information lost by compression. Small gap = state is nearly sufficient. Use this when something looks off to diagnose whether the bottleneck is compression vs predictor capacity.

### Additional probes (as needed)
- **Linear probes for dynamics properties (Eval C)**: Object velocity/direction (pseudo-GT from optical flow), scene change detection (binary), temporal ordering (which state came first).
- **Nearest-neighbor retrieval (Eval D)**: Do nearest neighbors in state space share dynamics or just appearance? Qualitative + quantitative (using activity labels).
- **Compression ratio Pareto (Eval F)**: Sweep state dimensionality, plot prediction quality. Find where you are on the information-compression tradeoff.

### What NOT to use
- Pixel-space reconstruction quality (rewards texture details irrelevant to planning)
- Single-frame classification like ImageNet probing (tests appearance, not dynamics)

## Step-by-Step Plan

I need to setup solid evals and a baseline for each!!!!

### Phase 1: Baseline (SSv2 only)
1. Extract and cache frozen DINO features for SSv2
2. Train recurrent encoder with MSE future prediction loss (single state, simplest architecture)
3. Measure: training loss, SSv2 linear probe accuracy
4. This is your baseline — all ablations compare against this

### Phase 2: Ablations on architecture/loss (SSv2, fast iteration)
5. Ablate state representation: cls token vs patches, state dimensionality
6. Ablate recurrent architecture: GRU vs TTT vs Linear TTT
7. Ablate loss: MSE vs contrastive vs combined. Add scene flow auxiliary loss.
8. For each ablation, measure: multi-horizon prediction decay curve (primary), SSv2 linear probe (secondary)
9. Pick best config

### Phase 3: Data scaling
10. Add Kinetics-700 — does more scene diversity improve the multi-horizon decay curve?
11. Train on combined SSv2 + Kinetics-700, re-run evals
12. Test on Epic-Kitchens-100 clips (longer videos) — does the decay curve hold for longer sequences?

### Phase 4: Stress test before stage 2
13. Run reconstruction gap analysis (Eval B) on Epic-Kitchens-100 to quantify information loss
14. Run dynamics probes (Eval C) to verify the state encodes motion, not just appearance
15. Run compression ratio Pareto (Eval F) to find the sweet spot for state size
16. If all look good: freeze stage 1 encoder, move to stage 2
