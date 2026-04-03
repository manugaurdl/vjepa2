# April 3 Meeting

models trained:  
action classification + next frame (h) — `zyvsy8gk`

next frame (h) — `qqm17c1k`

next frame *in dino space* (z) — `2ldiw9xk`

next frame PATCH TOKENS *in dino space* (z) — `e6esmgmu`

## Stage 1: Where We Are

- Proxy task: SSv2, 8 uniformly sampled frames, frozen DINO (vits14, 384-dim)
- Architecture: Additive RNN with predictive coding update
  - `Error = f(x_t) - W_pred(S_{t-1})`
  - `S_t = LN(S_{t-1} + W(Error))`
  - Loss = CE + 0.1 * L2(Error)
- Trained with action classification + next-frame prediction (pred_loss_weight=0.1, 100 epochs)
- Finding: predictive coding works better than sigmoid gating — easier to predict next frame than learn what to forget

## The Core Question

- The model isn't collapsed — it predicts next frames and classifies actions
- But: is the recurrent state learning **general dynamics** or just memorizing **SSv2 action prototypes**?
- We need eval that separates these two hypotheses

## Transfer Probe: UCF101

Freeze everything, train only a linear head on UCF101 (101 classes, 13K videos, 20 epochs, no augmentation, cached features).


| Model                                 | UCF101 Acc |
| ------------------------------------- | ---------- |
| DINO mean-pool (no temporal modeling) | **88.0%**  |
| DINO concat (8x384=3072-dim)          | **86.0%**  |
| RNN state (384-dim, frozen)           | **84.0%**  |


**Takeaway**: RNN state is worse than just averaging DINO features. The recurrent processing is trading general visual information for SSv2-specific features. The action classification loss is likely priming the state for SSv2's 174 action categories rather than general dynamics.

## Eval Setup Built This Week

1. **Transfer linear probe** (`eval_transfer.py`) — freeze encoder, train linear head on new dataset. Supports UCF101 and SSv2, with DINO mean-pool and concat baselines.
2. **OOD prediction decay** — compares pred_error_l2 curve on SSv2 vs UCF101 during training. Integrated into training loop, logs comparison plot to wandb (`ood_eval_csv` config).

where is the l2 baseline (if we predict mean dino feat) - for both clas and pathc

why not kinetics?

go beyond classification? 0

leWM

## What We've Tried

- **Prediction only (no action classification)**: collapsed — without the classification gradient, the learned prediction target space degenerates
- **Predict in frozen DINO space** (`predict_in_dino_space=True`): not working so far — high irreducible L2 loss. DINO features contain too much unpredictable detail (texture, lighting), model wastes capacity on noise rather than dynamics.

The dilemma: action classification prevents collapse but overfits the state to SSv2. Predicting in DINO space avoids that but doesn't converge well. Need a middle ground.

## Prediction in Dino space (SSv2, predict_in_dino_space=True)

Two baselines to contextualize pred_loss:

- **Predict global mean**: w_pred outputs the average DINO feature across all videos/frames. The dumbest predictor — measures total variance in the feature space.
- **Copy current frame**: w_pred perfectly copies the current frame as its prediction for the next. Exploits the fact that consecutive video frames are visually similar. This is the real baseline to beat.

All baselines computed on SSv2 val, skipping t=0 prediction (first frame prediction is always high since state is initialized to zeros).

- **Predict global mean**: predict the average DINO feature across all videos/frames. Dumbest predictor.
- **Predict local mean**: predict the average DINO feature of the current video. Stronger — captures per-video appearance.
- **Copy current frame**: predict the previous frame as the next. Exploits frame-to-frame smoothness. (L2 = X_t - X_{t-1})

Note: validation.pt has 168913 rows but only first 24777 are real val data (rest are zero-padded from old caching code). All baselines below computed on the correct 24777 val samples, skipping t=0.

### CLS token (S=1, D=384)


| Baseline               | L2      |
| ---------------------- | ------- |
| Predict global mean    | 1810    |
| Copy current frame     | 609     |
| Predict local mean     | 429     |
| **Model (100 epochs)** | **513** |


Model is 16% better than copy baseline but worse than local mean (429).

### Patch tokens (S=256, D=384)


| Baseline               | L2      |
| ---------------------- | ------- |
| Predict global mean    | 1804    |
| Copy current frame     | 1085    |
| Predict local mean     | 683     |
| **Model (100 epochs)** | **851** |


Model is 22% better than copy baseline but worse than local mean (683).

## How Well Do Models Understand Temporal Dynamics?

Beyond pred_loss, we need diagnostics that separate "learned dynamics" from "learned frame similarity."

> if model overwrites its state everytime, it becomes a last few frame copy-paste model

**1. Static vs Dynamic patch decomposition** (most direct, patch tokens only)

- Compute per-patch motion proxy (L2 between consecutive DINO frames), split into static (bottom 50%) and dynamic (top 50%) patches
- Compare model error vs copy baseline separately for each group
- Dynamics-aware model: improvement concentrated on dynamic patches. Smoothness-only model: equal improvement on both.

**2. Multi-step autoregressive rollout** (works for CLS and patches)

- At time t, roll out: w_pred(state_t) → w_pred(w_pred(state_t)) → ... without feeding new frames
- Compare error growth per horizon against ***copy baseline*** (frame t vs frame t+k)
- Dynamics model: error grows slower than copy baseline (extrapolates trajectories). Smoothness model: collapses to mean immediately.

++***copy baseline***++

The copy baseline at horizon k is just: how different is frame t from frame t+k in the dataset? No model involved — you just compute ||dino_feat[t] - dino_feat[t+k]||² averaged across videos, for k=1,2,3,... This gives you the "if I predicted frame t for all future frames, how bad would I be at each horizon?" curve. It's the natural extension of the copy-current-frame baseline to multiple steps. Your model's rollout error at horizon k should be compared against this. If the model's error grows slower than the copy baseline's, it's extrapolating dynamics rather than just repeating the last observation.

**3. Temporal shuffle test** (necessary but not sufficient)

- Evaluate on shuffled frame order. If pred_loss unchanged → model ignores temporal order entirely. If higher → temporal order matters, but could be dynamics OR just input autocorrelation helping state quality.

### Temporal Shuffle Results (SSv2 val, 3 random seeds averaged)

**Motivation**: The model beats the copy-current-frame baseline by 17-24%, but is that improvement from learning temporal dynamics or some other shortcut? We test this by shuffling the frame order at eval time. If the model learned dynamics (e.g., trajectory extrapolation), shuffling should break its predictions since temporal coherence is destroyed. If it just learned a slightly better static predictor, shuffling shouldn't matter much.

**Ratio** = shuffled pred_loss / normal pred_loss. 

Higher ratio = how bad the model is if frames are shuffled


| Model                                       | Normal | Shuffled | Ratio     | Copy Baseline |
| ------------------------------------------- | ------ | -------- | --------- | ------------- |
| `zyvsy8gk` (CE + pred, learned space)       | 2.23   | 2.29     | **1.03x** | —             |
| `2ldiw9xk` (pred only, CLS, dino space)     | 176.9  | 211.9    | **1.20x** | 620           |
| `e6esmgmu` (pred only, patches, dino space) | 851.5  | 1114.1   | **1.31x** | 1116          |


- Learned space model: temporal order barely matters (3%) — W_enc subspace doesn't encode real dynamics.
- CLS dino space: 20% increase — temporal order matters.
- Patch dino space: 31% increase, and shuffled loss (1114) ≈ copy baseline (1116) — **all improvement over copy comes from exploiting temporal order**.

### Copy Baseline Shuffle Sensitivity

To check whether the model's shuffle sensitivity comes from learned dynamics or just the task getting harder:


|                      | Copy Shuffle Ratio | Model Shuffle Ratio |
| -------------------- | ------------------ | ------------------- |
| CLS (dino space)     | 11.2x              | 1.20x               |
| Patches (dino space) | 1.46x              | 1.31x               |
| Pixel space          | 1.55x              | --                  |


**CLS**

- copy baseline gets **11x worse under shuffling** (random CLS pairs are very different), but the model only gets 1.2x worse — model is robust to shuffling, barely relies on temporal order despite the task getting dramatically harder.
- better for training - captures semantics - which keeps changing even if most tokens static
  - loses spatial structure (maybe)

**Patches** 

- copy baseline gets 1.46x worse, model gets 1.31x worse — nearly the same ratio. The model's shuffle sensitivity is al0most entirely explained by the task getting harder, not by the model having learned temporal structure.
  - captures spatial structure (similar to pixel space ~ both get around 1.5x shuffle ratio), but **gradient dilution problem** --> once model learns to capture scene level info early in the training, loss = 0 for static tokens (N_s) , however loss over dynamic tokens (N_d << N_s) still get scaled down by N_s. 



**Interpreting the gap between copy and model shuffle ratios:**

- DINO CLS space has 11.2x shuffle sensitivity — a useful **upper bound** — it tells you how much temporal signal exists in CLS space that a good predictor could exploit. 

- DINO patches (1.46x) behave like pixels (1.55x) - most pixels static
- Neither model shows strong temporal dynamics yet. The patch model's higher shuffle ratio (1.31x) tracks its copy baseline (1.46x), not better dynamics learning.

**4. Video reversal test** (weaker than expected)

- Reversed video preserves frame similarity but flips causal direction. However, the Bayesian update rule adapts — after 2-3 reversed frames, the state accumulates reversed dynamics and predicts accordingly. So a good dynamics model performs well on reversed video too, making this test less informative.

## Next Steps

- Find a collapse prevention mechanism that doesn't bias the state toward SSv2 (EMA target encoder? variance regularization?)
- Run **OOD decay curve** to diagnose whether the state has memorized SSv2-specific temporal patterns
- **SSv2 linear probe** on future prediction-only checkpoints once collapse is solved

