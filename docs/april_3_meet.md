# April 3 Meeting

### Models trained:

1. action classification + next frame (h) — `zyvsy8gk`
2. next frame (h) — collapsed

3a. next frame *in dino space* (z) — `2ldiw9xk`
3b. next frame PATCH TOKENS *in dino space* (z) — `e6esmgmu`

---

## Stage 1: Where We Are

### Action Classification + Next frame latent prediction (h=W(DINO_cls))

- Proxy task: SSv2, 8 uniformly sampled frames, frozen DINO (vits14, 384-dim)
- Architecture: Additive RNN with predictive coding update
  - `Error = f(x_t) - W_pred(S_{t-1})`
  - `S_t = LN(S_{t-1} + W(Error))`
  - Loss = CE + 0.1 * L2(Error)
- Trained with action classification + next-frame prediction (pred_loss_weight=0.1, 100 epochs)
- Finding: predictive coding works better than sigmoid gating — easier to predict next frame than learn what to forget

## The Core Question

- The model isn't collapsed — it predicts next frames and classifies actions
- But: is the recurrent state learning **general dynamics** or  **overfitting to SSV2 + action recognition**?
  - we know its not overfitting to ssv2 (Q: what about non action recognition tasks)
- We need eval that separates these two hypotheses

---

## Transfer Probe: UCF101

#### Does action classification generalize beyond SSV2?

Freeze everything, train only a linear head on UCF101 (101 classes, 13K videos, 20 epochs, no augmentation, cached features).

*Table 1*


| Model                                            | UCF101 Acc |
| ------------------------------------------------ | ---------- |
| DINO mean-pool (no temporal modeling)            | **88.0%**  |
| DINO concat (8x384=3072-dim)                     | **86.0%**  |
| RNN CLS, DINO-space pred (`2ldiw9xk`)            | **85.4%**  |
| RNN CLS, CE+pred, learned space (`zyvsy8gk`)     | **84.0%**  |
| RNN Patches, DINO-space pred (`e6esmgmu`)        | **81.7%**  |
| RNN Patches, CE+pred, learned space (`tj9x820q`) | **78.3%**  |


### Takeaways

- All RNN states trail DINO mean-pool (88%)
- CLS > patches consistently (~4-7 pts)

 *maybe above two change with better W_pred*

- DINO-space > JEPA + action classification: CE loss overfits to SSV2
- Need non-action-recognition eval to test that

### Takeaways

- dino-space better than JEPA right now - maybe changes with SigReg etc.
- Table 4 suggests, patch prediction better preserves temporal order
  - maybe UCF measures general vision features not temporal (can explain why mean pool > concat)

---

## What We've Tried

- **Prediction only (no action classification)**: collapsed — without the classification gradient, the learned prediction target space degenerates
- **Predict in frozen DINO space** (`predict_in_dino_space=True`): not working so far — high irreducible L2 loss. DINO features contain too much unpredictable detail (texture, lighting), model wastes capacity on noise rather than dynamics. (Q:need to do this)

---

## Prediction in Dino space (SSv2)

All baselines computed on SSv2 val, skipping t=0 prediction (first frame prediction is always high since state is initialized to zeros).

- ~~**Predict local mean**: predict the average DINO feature of the current video. Stronger — captures per-video appearance.~~
- **Copy current frame**: predict the previous frame as the next. Exploits frame-to-frame smoothness. (L2 = X_t - X_{t-1})

> beyond L2 which measures how good recurrent state and W_pred are; important baseline = downstream transfer (using recurrent state vs concat dino vs meanpool dino)

### CLS token (S=1, D=384)

*Table 2*


| Baseline                             | L2        |
| ------------------------------------ | --------- |
| Copy current frame                   | 609       |
| Causal Transformer `ud2ncxlq`        | **100.1** |
| RNN `2ldiw9xk`                       | 176.9     |


### Patch tokens (S=256, D=384)

*Table 3*


| Baseline                             | L2      |
| ------------------------------------ | ------- |
| Copy current frame                   | 1085    |
| Causal Transformer (concat baseline) | **783** |
| RNN                                  | 851     |


### Mean-pooled patch tokens (S=1, D=384)

*Table 3b*

| Baseline                             | L2    |
| ------------------------------------ | ----- |
| Copy current frame                   | 162.8 |
| Causal Transformer                   | tbd   |
| RNN                                  | tbd   |

Copy baseline 162.8 — much lower than CLS (609) and raw patches (1085). Mean-pooling averages 256 spatial locations, making consecutive frames nearly identical. Very little temporal signal — easy to predict but potentially poor training signal for dynamics.

**Takeaways:**

- Causal transformer CLS (`ud2ncxlq`) — upper-bound baseline with full attention over past frames, no state bottleneck. At 27 epochs it matches the RNN, suggesting the RNN state isn't losing much information.
- Causal transformer PATCH (`r55x2lcn`) — The state bottleneck isn't the issue — w_pred expressiveness or the prediction task itself is the limit.
- both models better than copy-frame baseline
- **Can't outperform self-attention across patch tokens (need more expressive state update)**
  - For patches (seq length 2048), the causal transformer can explicitly model "patch (5,8) at frame 3 moved to (5,9) at frame 4, so predict (5,10) at frame 5." 
  - RNN: (8,256) is sufficient capacity; need more expressive state update and W_pred

---

## How Well Do Models Understand Temporal Dynamics?

Beyond pred_loss, we need diagnostics that separate "learned dynamics" from "learned frame similarity."

> copy-paste baseline or a better exp. moving average gets decent l2 loss

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

---

**3. Temporal shuffle test** (necessary but not sufficient)

- Evaluate on shuffled frame order. 
- If pred_loss :
  - unchanged → model ignores temporal order entirely. 
  - higher → temporal order matters. how much dynamics modelled not sure

### Temporal Shuffle Results (SSv2 val, 3 random seeds averaged)

**Motivation**: 

The model beats the copy-current-frame baseline by 17-24%, but is that improvement from learning temporal dynamics or a slightly better exp. moving average (for ex, static bg modelled of last 3 frames)

We test this by shuffling the frame order at eval time:

- If the model learned dynamics (e.g., trajectory extrapolation), shuffling should break its predictions since temporal coherence is destroyed. However, If it has just learned a slightly better static predictor, shuffling shouldn't matter much.

**Ratio** = shuffled pred_loss / normal pred_loss. 

**Higher ratio = how bad the model is if frames are shuffled**

*Table 4*


| Model                                          | Normal | Shuffled | Ratio     | Copy Baseline |
| ---------------------------------------------- | ------ | -------- | --------- | ------------- |
| `zyvsy8gk` (CE + pred, CLS, learned space)     | 2.23   | 2.29     | **1.03x** | —             |
| `tj9x820q` (CE + pred, patches, learned space) | 3.80   | 4.29     | **1.13x** | —             |
| `2ldiw9xk` (pred only, CLS, dino space)        | 176.9  | 211.9    | **1.20x** | 620           |
| `e6esmgmu` (pred only, patches, dino space)    | 851.5  | 1114.1   | **1.31x** | 1116          |


- Learned space model: temporal order barely matters (3%) — W_enc subspace doesn't encode real dynamics (overfit to action recognition?)

To interpret the dino-space models, we also measure how much the copy baseline degrades under shuffling:

*Table 5: Copy Baseline Shuffle Sensitivity*


|                      | Copy Shuffle Ratio | Model Shuffle Ratio |
| -------------------- | ------------------ | ------------------- |
| CLS (dino space)     | 11.2x              | 1.20x               |
| Patches (dino space) | 1.46x              | 1.31x               |
| Pixel space          | 1.55x              | --                  |


Note: copy ratio measures a data property (how much worse is frame similarity between random pairs vs consecutive pairs). Model ratio measures a model property (how much the model relied on temporal order).

---

### CLS

The model learns two things:

1. **Multi-frame aggregation** (order-independent) — combining information from all frames gives a much better prediction than any single frame. This survives shuffling: shuffled model (211.9) is still far better than copy baseline (620).
2. **Temporal dynamics** (order-dependent) — the 20% degradation from shuffling (176.9 → 211.9).

The CLS copy baseline has 11.2x shuffle ratio because random CLS pairs are very different (CLS summarizes the whole scene), so copying a random frame is terrible. But the model doesn't care — it's not relying on "the last frame," it's using all of them.

### Patches

Shuffled model (1114) ≈ copy baseline (1116). When you destroy temporal order, the model becomes exactly as bad as copy-paste.

No benefit of having a *running summary* (as in CLS) — patches are mostly static, so one frame is already a sufficient predictor. DINO patches (1.46x) behave like pixels (1.55x).

All of the model's advantage over copy (851 vs 1085) is destroyed by shuffling, meaning it came entirely from temporal order.

**Q: If I just compute L2 difference on dynamic patch tokens, I should get really high copy shuffle ratio → is this why CLS is so high?**

---

### CLS or Patch Prediction?

- **CLS** gives a cleaner training signal (no gradient dilution, every frame contributes meaningfully) 
- does it capture dyanmics (motion structure) or just semantic differences?  
- Throws away spatial information that stage 2 likely needs for forecasting.

**Patches** have the right inductive bias — spatial + temporal, 

- gradient dilution

For stage 1 (state expressive enough for stage 2 forecasting), patches are better aligned. The question is how to fix gradient dilution:

1. **Motion-weighted loss**: compute per-patch motion proxy, upweight dynamic patches in L2 loss. Directly targets the problem.
2. **Predict only residuals**: predict `x_{t+1} - x_t` instead of raw features. Static patches become ~zero, dynamic patches become the entire signal. Eliminates static patch problem without adding hyperparameters.
3. **CLS as auxiliary**: train on patch prediction (primary) + CLS prediction (auxiliary). CLS loss keeps gradients flowing through the state even when patch loss stagnates on static tokens.

---

## Static vs Dynamic patch decomposition**

Does the patch model's improvement over copy come from learning dynamics or just smoothness?

1. Compute per-patch motion score from GT features: average frame-to-frame L2 per spatial position per video
2. Median split per video: **top 50% = dynamic, bottom 50% = static**
3. Copy baseline error (`||x_t - x_{t-1}||²`) averaged separately over each group
4. Model `pred_error_l2` (B, T, S), skip t=0, averaged separately over each group
5. Compare: `(copy_err - model_err) / copy_err` per group

Also: copy shuffle ratio on dynamic patches only — if high (close to CLS's 11.2x), CLS is just tracking dynamic content. If low, CLS captures something else.

Script: `root/evals/static_dynamic_decomposition.py`

**Results** (`e6esmgmu`, pred only, patches, DINO space):


|                 | Copy Baseline | Model  | Improvement |
| --------------- | ------------- | ------ | ----------- |
| Dynamic patches | 1430.6        | 1077.0 | **24.7%**   |
| Static patches  | 746.9         | 625.9  | **16.2%**   |


- Improvement concentrated on dynamic patches (1.5x ratio) → model learned dynamics, not just smoothing.
- `tj9x820q` **(CE + pred, patches, learned space)**: model error dynamic = 4.2, static = 3.0  Dynamic patches harder to predict even in learned space  (can't have copy baseline in latent space)

### Dynamic-only copy shuffle ratio for patches= **1.40x** (vs CLS 11.2x)

- Thus, CLS captures something beyond patch-level motion (visual/semantic shifts, not just dynamics).

---

**4. Video reversal test** (weaker than expected)

- Reversed video preserves frame similarity but flips causal direction. However, the Bayesian update rule adapts — after 2-3 reversed frames, the state accumulates reversed dynamics and predicts accordingly. So a good dynamics model performs well on reversed video too, making this test less informative.

---

## To Do

### 1. Make prediction work without action classification

Two parallel approaches:

**a) JEPA-style learned-space prediction (h = W(DINO)) + SigREG**

Currently, prediction-only in learned space collapses — W_enc degenerates to make prediction trivial. 

- SigReg
- EMA/ stop gradient

**b) More expressive W_pred for DINO-space prediction**

- recurrent state not a bottleneck (table 2)
- expressive W_pred should help bridge patch tokens performance to causal transformer
- also, make it as good a mean pooled embeddings (table 1)

### 2. Diagnostics on existing models

**b) OOD prediction decay curve (tbd)** 

Does the recurrent state capture general dynamics or SSv2-specific temporal patterns?

- Run inference on SSv2 val and UCF101 test with the same checkpoint
- Compare per-timestep pred_error_l2 curves across distributions
- Similar decay shape on both → general dynamics. Much higher/flatter on UCF101 → memorized SSv2 patterns.
- Pipeline exists: `eval_transfer.py --mode decay`. Run on DINO-space checkpoints (`2ldiw9xk`, `e6esmgmu`).

**c) Multi-step autoregressive rollout**

Can the model extrapolate trajectories, or does it collapse to the mean without new frames?

- At time t, autoregressively roll out: w_pred(state_t) → w_pred(w_pred(state_t)) → ... for k steps without feeding new frames
- Compare error growth per horizon against the copy baseline (||dino_feat[t] - dino_feat[t+k]||²)
  - **If model error grows slower than copy** → extrapolating dynamics. 
  - **If model collapses to mean immediately** → no trajectory learning.
- Works for both CLS and patch models.

### 2. Ultimate eval - long horizon chunk prediction (i.e stage 2)

- eval A in stage 1 md

### ask saining

- leWM --> zero shot planning is nice, but they train and eval on sims
- long horizon data hard - ego4d; sims
- generative or contrastive
  - MSE --> futures multimodal 
  - pixels vs latent

---

