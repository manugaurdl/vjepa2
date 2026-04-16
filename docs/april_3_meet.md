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
- Table 4 (corrected): CLS and patches both lose all their edge over copy when frames are shuffled — the old "CLS aggregation survives shuffling" claim was an artifact of the padding bug
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


| Baseline                      | L2      |
| ----------------------------- | ------- |
| Copy current frame            | 609     |
| Causal Transformer `ud2ncxlq` | **517** |
| RNN `2ldiw9xk`                | 513     |


### Patch tokens (S=256, D=384)

*Table 3*


| Baseline                      | L2      |
| ----------------------------- | ------- |
| Copy current frame            | 1089    |
| Causal Transformer `r55x2lcn` | **789** |
| RNN `e6esmgmu`                | 851     |


### Mean-pooled patch tokens (S=1, D=384)

*Table 3b*


| Baseline                      | L2        |
| ----------------------------- | --------- |
| Copy current frame            | 162.8     |
| Causal Transformer `1hmdjkmc` | **136.4** |
| RNN `k5qezvem`                | 139.3     |


Copy baseline 162.8 — much lower than CLS (609) and raw patches (1089). Mean-pooling averages 256 spatial locations, making consecutive frames nearly identical. Very little temporal signal — easy to predict but potentially poor training signal for dynamics.

RNN `k5qezvem` (last.pt, epoch 26) improves over copy by **14.4%** (162.8 → 139.3). Trained with `--meanpool_patches --encoder.rnn.predict_in_dino_space --no-action_classification` — loss is `(w_pred(state) - meanpool_DINO[t])²` in raw DINO space, directly comparable to copy baseline units.

Causal Transformer `1hmdjkmc` (best.pt, epoch 13) improves over copy by **16.2%** (162.8 → 136.4) — modest edge over the RNN. Shuffle ratio is 1.62x (vs RNN not yet shuffle-tested in meanpool space) — the transformer relies more heavily on frame order. Trained with `--meanpool_patches --encoder.type causal_transformer --no-action_classification` for 44 epochs total (killed early); best.pt from epoch 13 by pred_loss.

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

**Why this baseline is the right one for AR rollout:** The copy curve `||x_t - x_{t+k}||²` is a pure data property — it tells you how fast the dataset itself drifts, which is the error any non-extrapolating predictor (constant, EMA, smoother) is stuck with. An EMA rolled out autoregressively has nothing to integrate once frames stop coming, so it collapses to a constant and tracks this drift curve exactly.

Your model sits on a spectrum between two anchors:

- **Oracle** (0 L2): perfect dynamics, knows the exact future.
- **Copy/EMA** (the drift curve): no dynamics, outputs a constant.

A real dynamics model lives in between — error grows with `k` (compounding + stochastic futures), but slower than the drift curve. The gap between model and copy at horizon `k` is literally "how much trajectory extrapolation the model bought you `k` steps out."

### AR rollout results (evalID: `autoregressive_rollout`, `t_ctx=4`, K=4)

Script: `root/evals/autoregressive_rollout.py`. Context = first 4 frames; then iterate `w_pred` 4 times with no new input. RNN + DINO-space only (learned-space has no comparable unit; causal transformer rollout requires append-token mechanics and is out of scope).

**Baselines**:

- **Copy** (zero velocity): `x̂_{t+k} = x[t_ctx-1]`. Drift floor — what any constant/EMA predictor is stuck with.
- **Linear** (constant velocity): `x̂_{t+k} = x[t_ctx-1] + k·(x[t_ctx-1] - x[t_ctx-2])`. "Dumb dynamics" — velocity estimated from the last two observed frames. A model that can't beat linear hasn't learned anything beyond first-order motion.

*Table 7: Per-horizon L2 (sum over D, mean over S for patches)*


| Model                            | k   | copy | linear | model   | model/copy         |
| -------------------------------- | --- | ---- | ------ | ------- | ------------------ |
| `2ldiw9xk` (CLS, pred, DINO)     | 1   | 743  | 1895   | **607** | **0.82x** ← extrap |
|                                  | 2   | 993  | 4923   | 3304    | 3.33x ← collapse   |
|                                  | 3   | 1134 | 9360   | 3964    | 3.50x ← collapse   |
|                                  | 4   | 1204 | 15124  | 4742    | 3.94x ← collapse   |
| `e6esmgmu` (Patches, pred, DINO) | 1   | 1271 | 3380   | **958** | **0.75x** ← extrap |
|                                  | 2   | 1604 | 8622   | 4487    | 2.80x ← collapse   |
|                                  | 3   | 1767 | 16243  | 7686    | 4.35x ← collapse   |
|                                  | 4   | 1856 | 26219  | 15683   | 8.45x ← collapse   |


**Takeaways:**

- Both models beat copy at k=1 — expected, that's exactly what they were trained on (`w_pred(state)` predicts the next real frame).
- From k=2 onward both models are **strictly worse than copy**, by 3–8x. Iterating `w_pred` on its own output diverges from the DINO manifold fast.
- Neither model has a stable attractor under self-application of `w_pred`. This is the "smart 1-step predictor, not a dynamics model" failure mode.
- Linear extrapolation in raw DINO space is useless (huge velocities) — it's in the table as a sanity check, not as a competitive baseline.

---

**3. Temporal shuffle test** (necessary but not sufficient — **inconclusive** for learned-dynamics vs smart-EMA)

- Evaluate on shuffled frame order.
- If pred_loss :
  - unchanged → model ignores temporal order entirely.
  - higher → temporal order matters. how much dynamics modelled not sure
- **Why it cannot distinguish dynamics from a smart EMA:** an EMA is also order-sensitive — the most recent frame is weighted highest, so shuffling changes which frame dominates and pred_loss degrades. Ratio > 1 just proves the model is not permutation-invariant; it does NOT imply trajectory learning. The autoregressive rollout test is the one that actually separates the two hypotheses.

### Temporal Shuffle Results (SSv2 val, 3 random seeds averaged)

**Motivation**: 

The model beats the copy-current-frame baseline by 17-24%, but is that improvement from learning temporal dynamics or a slightly better exp. moving average (for ex, static bg modelled of last 3 frames)

We test this by shuffling the frame order at eval time:

- If the model learned dynamics (e.g., trajectory extrapolation), shuffling should break its predictions since temporal coherence is destroyed. However, If it has just learned a slightly better static predictor, shuffling shouldn't matter much.

**Ratio** = shuffled pred_loss / normal pred_loss. 

**Higher ratio = how bad the model is if frames are shuffled**

*Table 4* (corrected 2026-04-08 after fixing the `validation.pt` padding bug — see `docs/exp_progress.md`; old DINO-space numbers were diluted ~7x by ~144k zero-padded rows)


| Model                                          | Normal | Shuffled | Ratio     |
| ---------------------------------------------- | ------ | -------- | --------- |
| `zyvsy8gk` (CE + pred, CLS, learned space)     | 3.70   | 4.13     | **1.12x** |
| `tj9x820q` (CE + pred, patches, learned space) | 3.59   | 4.08     | **1.14x** |
| `2ldiw9xk` (pred only, CLS, dino space)        | 513    | 752      | **1.47x** |
| `e6esmgmu` (pred only, patches, dino space)    | 851    | 1114     | **1.31x** |


- Learned space model: temporal order barely matters (3%) — W_enc subspace doesn't encode real dynamics (overfit to action recognition?)

To interpret the dino-space models, we also measure how much the copy baseline degrades under shuffling:

*Table 5: Copy Baseline Shuffle Sensitivity* (corrected 2026-04-08)


|                       | Copy (normal) | Copy (shuffled) | Copy Shuffle Ratio | Model Shuffle Ratio |
| --------------------- | ------------- | --------------- | ------------------ | ------------------- |
| CLS (dino space)      | 609           | 1004            | 1.65x              | 1.47x               |
| Patches (dino space)  | 1089          | 1588            | 1.46x              | 1.31x               |
| Meanpool (dino space) | 163           | 277             | 1.70x              | 1.53x               |
| Pixel space           | —             | —               | 1.55x              | --                  |


Note: copy ratio measures a data property (how much worse is frame similarity between random pairs vs consecutive pairs). Model ratio measures a model property (how much the model relied on temporal order). The earlier CLS copy ratio of **11.2x** did not reproduce — it was a bug in the original computation. CLS and patches have similar data smoothness (~1.5x).

---

### CLS (rewritten after padding-bug fix)

The old narrative was: "the CLS model beats copy by a lot because it aggregates multiple frames, and this advantage survives shuffling." That narrative was built on the diluted 176.9/211.9 numbers and is **wrong**.

Corrected picture:

- Normal model (513) beats copy (609) by only ~16%.
- **Shuffled model (752) is worse than copy baseline (609).** Every bit of the CLS model's edge over copy is order-dependent.
- There is no "order-independent aggregation advantage" to speak of. Under shuffled order, the model is strictly worse than just copying the previous (shuffled) frame.

So CLS behaves qualitatively the same as patches now: the model's win over copy comes entirely from temporal order, and destroying that order collapses it.

### Patches

Shuffled model (1388) > copy baseline (1085). Same story as CLS: the model's advantage over copy (1082 vs 1085 — essentially none to begin with in absolute terms, but the training loss at this checkpoint is near copy) is destroyed by shuffling.

DINO patches (1.46x) behave like pixels (1.55x) in terms of data smoothness.

All of the model's advantage over copy is order-dependent in both CLS and patch space.

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

Also: copy shuffle ratio on dynamic patches only — originally framed as "if close to CLS's 11.2x, CLS is just tracking dynamic content." Stale since the 11.2x figure did not survive the padding-bug fix (corrected CLS copy ratio is 1.65x). The static/dynamic decomposition still stands on its own.

Script: `root/evals/static_dynamic_decomposition.py`

**Results** (`e6esmgmu`, pred only, patches, DINO space):

table 6:


|                 | Copy Baseline | Model  | Improvement |
| --------------- | ------------- | ------ | ----------- |
| Dynamic patches | 1430.6        | 1077.0 | **24.7%**   |
| Static patches  | 746.9         | 625.9  | **16.2%**   |


- Improvement concentrated on dynamic patches (1.5x ratio) → model learned dynamics, not just smoothing.
- `tj9x820q` **(CE + pred, patches, learned space)**: model error dynamic = 4.2, static = 3.0  Dynamic patches harder to predict even in learned space  (can't have copy baseline in latent space)

### Dynamic-only copy shuffle ratio for patches= **1.40x** (CLS comparison dropped — corrected CLS copy ratio is 1.65x, not 11.2x)

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

## Evaluating Model's Ability to Extrapolate

The ability to extrapolate/forecast is a great way to test if the model captures dynamics. AUTOREGRESSIVE EVAL doesn't make sense for our model as we make future prediction using state S i.e --> creating State S at timestep T requires ground-truth frame x_t

#### What is another good way to evaluate if the model captures dynamics? Or what arch change can I make so autoregressive eval makes sense.?

# CONVO WITH CLAUDE:

---

## A. Arch changes that make AR rollout meaningful

### A1. Add an explicit forward/transition model `f(state) → next_state`

This is the DreamerV3 / PlaNet / world-model move. Decouple two things the current design conflates:

- **Posterior update** (observation): `state_t = g(state_{t-1}, x_t)` — your existing surprise update.
- **Prior transition** (imagination): `ŝ_{t+1} = f(state_t)` — new, a learned state→state map with no x input.
- **Decoder**: `x̂_t = w_pred(state_t)` — same as now.

**Training**: at each step, compute both the posterior `state_t` (using real `x_t`) and the prior `ŝ_t = f(state_{t-1})`. Add a KL/L2 loss pushing `ŝ_t` toward `state_t`. This forces `f` to learn "how the state evolves in time" independent of observations.

**Why this unbreaks AR rollout**: rollout becomes `ŝ_{t+k} = f^k(state_t)`, with `x̂_{t+k} = w_pred(ŝ_{t+k})`. `f` is trained to match real state trajectories, so iterating it stays on-manifold instead of drifting to a fixed point like `w_pred^k` does.

This is the cleanest fix and the standard recipe in world-model literature. Minimal cost: one extra MLP + one extra loss term.

### A2. Multi-horizon prediction heads

No transition model, but train `w_pred_k: state → x_{t+k}` for `k=1..K` jointly. Each horizon gets its own head; the state is forced to carry multi-step info. AR rollout is replaced by **parallel multi-horizon decoding from a single state**.

Cheaper than A1 (no imagination loss), but less principled — you're paying for every horizon with a separate head, and you can't extrapolate beyond the trained `K`.

---

## B. Evals that test dynamics **without rollout**, on existing checkpoints

### B1. Multi-horizon linear probe from state

Freeze the trained state. Train linear heads `state_t → x_{t+k}` for `k=1..4` on train split, eval on val. Compare per-horizon probe L2 to the copy baseline `||x_t - x_{t+k}||²`.

**What it measures**: "is multi-step future linearly decodable from the state?" If state_t only encodes the current frame (smart EMA), the probe for `k=1` works but `k≥2` collapses to copy. If state_t encodes a trajectory, the probe should beat copy for several horizons.

**Why this is fair to your arch**: the model was trained with a single-step head, but if the state *contains* multi-step information, a cheap linear probe will find it. This tests the state, not `w_pred`. Minimal cost, no retraining.

### B2. Velocity / displacement probe

Linear probe `state_t → (x_{t+1} - x_t)` — can the state linearly predict the *direction of change*, not just the current frame? Copy baseline predicts zero displacement, so the probe only wins if the state knows which way things are moving.

Stronger version: probe `state_t → (x_{t+k} - x_t)` for several `k`. Tests whether the state encodes "motion vectors" at different horizons.

### B3. Action classification from state *delta*

Train linear probe on `(state_t - state_{t-k})` → action class. SSv2 actions are defined by motion ("pushing something from left to right"), so action-from-delta accuracy is a direct test of whether the state transition encodes motion semantics. If `state_t - state_{t-1}` is just noise, this probe fails; if it's a coherent motion representation, it wins.

### B4. Frame-order cloze / ranking

Given state at some intermediate `t`, score the true `x_{t+1}` against distractors (other frames in the same video at wrong timesteps, or frames from other videos). Measure top-1 ranking accuracy. This bypasses L2-in-noisy-DINO-space entirely — you only need the model to *rank* correctly, which is a much easier signal to see and compare against a copy baseline (copy ranks by similarity to `x_t`).

---

## My recommendation

**Short-term, zero retraining**: run **B1 (multi-horizon linear probe)** on `2ldiw9xk` and `e6esmgmu`. This directly answers "does the state carry multi-step info?" without needing your architecture to support rollout. If the probe beats copy at k=2,3,4, the state does encode dynamics — the current `w_pred` just can't extract it in an iterable way. If the probe *doesn't* beat copy, the state genuinely only encodes the current frame and you need A1.

**Medium-term**: implement **A1 (explicit forward model)**. This is the only principled way to make your stated goal — "compact state good enough for long-horizon forecasting (stage 2)" — actually testable. Without a transition model, "long-horizon forecasting" has no mechanism. A1 is one extra MLP and one extra loss term, and it makes AR rollout the natural test again.

The combination is tight: B1 tells you whether the current state is worth rolling forward; A1 gives you the rollout mechanism if it is.

---

The core distinction: B1 is a probe, A2 is an intervention

They measure different things and are not redundant.

- B1 (linear probe on frozen state) — tests what's in the state. It tells you whether the state already carries multi-step info, regardless of how you got there.
- A2 (train with multi-horizon heads) — changes what goes into the state via extra gradient pressure. It's an intervention on training, not a measurement.

The built-in A2 heads conflate two things: "did my state get better?" and "did my decoder get better?" B1 decouples those. That's exactly why you still want B1 even after training A2.

What B1 on the current model actually buys you

Running B1 on 2ldiw9xk / e6esmgmu answers: "did multi-step info leak into the state as a byproduct of k=1 training?"

To predict x_{t+1} well, the state probably needs to encode "what's happening" — object identity, positions, motion vectors — and that representation may linearly decode further ahead with no retraining.

Three clean outcomes:

1. B1 beats copy at all horizons → the state is fine, it already carries trajectory info. The AR rollout failure was a decoder problem, not a state problem. → skip A2 entirely, go straight to A1 (forward model) since your bottleneck is the rollout mechanism, not the representation.
2. B1 only beats copy at k=1 → the k=1 training objective genuinely produces a myopic state. → A2 is justified (or A1 + multi-horizon loss combined).
3. B1 beats copy at k=1, 2 but collapses at 3, 4 → partial. A2 might push further; worth trying. Also gives you a concrete baseline to beat.

Without B1, you don't know which world you're in. Training A2 when outcome (1) is true wastes compute on the wrong problem.

What B1 on an A2-trained model buys you (if you train A2)

A2's built-in heads are not a substitute for B1 post-A2.

Three separate things you can only get by running B1 on the A2 model:

- State-vs-decoder decomposition: fresh linear probes on frozen A2 state, compared against B1 on the k=1 model at matching horizons. Equal scores ⇒ A2's gains came from better decoders, state unchanged. A2 probe > k=1 probe ⇒ A2 actually improved the representation.
- Horizon generalization: if A2 was trained for k=1..4, probe at k=5, 6, 7. Built-in A2 heads can't do this — they only exist for trained horizons. Tells you whether the state encodes "trajectories" or just the specific horizons it was supervised on.
- Head calibration check: B1 probe accuracy vs A2 built-in head accuracy at the same horizon. If the linear probe matches the built-in head, the heads are well-trained. If it's much better, the heads are under-fit.

The methodical order

1. B1 on current models first. No retraining. Fast. Tells you which world you're in.
2. Decide next step based on B1 outcome:
  - Outcome 1 → A1 (forward model) — state is fine, rollout mechanism is broken.
  - Outcome 2 → A2 (multi-horizon training) — state is myopic.
  - Outcome 3 → A2 + A1 combined — partial state, also broken rollout.
3. After any A2 training, run B1 on the new model and compare to B1 on the old one. This is the step that isolates state improvement from decoder improvement and makes the A2 result interpretable.
4. If you end up training A1, AR rollout itself becomes the natural eval again — no probes needed, the forward model gives you meaningful rollouts.

Short answer to your specific questions

- "Does B1 make sense if model not trained with A2?" Yes — that's precisely when it's most informative, because it tells you whether A2 is even needed.
- "Is the current model a good baseline for A2?" Yes, and the comparison B1@current vs B1@A2 is the cleanest way to show A2 did something beyond fitting better heads.
- "If I train A2, do I still need B1?" Yes — for state-vs-decoder decomposition and horizon generalization, which the built-in heads can't give you.

Bottom line: B1 first (cheap, decisive), then decide A1/A2 based on outcome, then B1 again after any training to cleanly attribute the gains.

### Multi-horizon probe eval

- Given batch size = 32, T=8, dim = : the encoder spits outs (32, 8, D) - lets consider CLS for simplicity

**Step** **1**. **Run** **the** **encoder** **on** **all** **SSv2** **videos.** For each video (N=168k train, 24k val), run the RNN on all 8 frames. For each frame x_t, you get a state_t; a 384 dim vector.

**Step** **2**. **Collect** **(input,** **target)** **pairs** **for** **ridge** **regression.** For each horizon k (1 through 4), loop over valid time steps t=1 to T−k−1.

Same target, different inputs. For k=1 with T=8, that's 6 valid t values per video × 168k videos ≈ 1M training rows.

**Step** **3** **—** **Fit one** **linear** **probe per horizon** **(closed-form,** **on** **train).** For each k, solve:                                                                           

W* = argmin || ***input*** @ W − x_{t+k}||² + λ||W||²

Here input is taken from timestep t (can be *recurrent state* or *dino ground truth* feature), and used to predict the DINO feature at timestep=T+K

Solved in one shot via W = (XᵀX + λI)⁻¹ XᵀY — no SGD, no epochs, no hyperparameters to tune. The accumulators XᵀX and XᵀY are streamed in fp64, so this is numerically stable even over 1M rows.                                                            

**Step** **4** — **Evaluate** **on** **val.** Same loop over (video, t) pairs on the val split. Compute three errors per sample:


| Column          | Formula                        | What it measures                                                                                                                                                       |
| --------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| copy-last       | L2(x_t, x_{t+k})               | **drift baseline:** "How much do frames drift over k steps?" A predictor that just outputs the current frame gets this error.                                          |
| raw-DINO_linear | L2(x_t @ W_raw, x_{t+k})       | Best linear prediction from the current frame alone, no history. Captures any static/structural regularity (e.g. background persists, objects stay roughly in place). |
| state           | L2(state_t @ W_state, x_{t+k}) | Best linear prediction from the RNN's state after seeing frames 0..t. Has access to temporal history — could encode velocity, trend, periodic motion.                  |


Average each over all val (video, t) pairs → the numbers in the table.

**Derived columns (additive decomposition):**

Total improvement over copy decomposes cleanly into two parts:

```
(copy - state)  =  (copy - raw_dino)  +  (raw_dino - state)
 total improvement   structural gain       temporal gain
```


| Metric           | Formula            | What it measures                                                                                                                                                                 |
| ---------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| copy - state     | `copy - state`     | **Total improvement**: how much better the state probe is than copying last encoded frame                                                                                        |
| copy - raw_dino  | `copy - raw_dino`  | **Structural gain**: what a linear projection of the current frame buys you over just copying last encoded frame (static regularity, background persistence)                     |
| raw_dino - state | `raw_dino - state` | **Temporal gain**: what the encoder's temporal modeling adds on top on a linear projection of last frame.Positive means the state carries information beyond the current frame. |


**Table 7: CLS Results** 


| k   | copy   | raw_dino | state | copy-stateTOTAL GAIN | copy-raw_dinoLIN PROJ GAIN | dino_raw - stateTEMPORAL GAIN |
| --- | ------ | -------- | ----- | --------------------- | --------------------------- | ------------------------------ |
| 1   | 632.3  | 562.1    | 524.0 | 108.2                 | 70.1                        | 38.1                           |
| 2   | 925.3  | 779.6    | 730.2 | 195.1                 | 145.7                       | 49.4                           |
| 3   | 1123.1 | 912.6    | 860.4 | 262.7                 | 210.5                       | 52.2                           |
| 4   | 1238.3 | 986.8    | 936.0 | 302.4                 | 251.6                       | 50.8                           |


- All three errors grow with k.
- `copy - raw_dino` (structural gain) grows with k: a linear projection compensates for more drift at longer horizons.
- `raw_dino - state` (temporal gain) is ~38–51 across all k — modest but consistent and positive at every horizon. The state isn't myopic.¯¯

**Table 7: Patch Results (**`e6esmgmu`**):** token-wise probe, shared W: 384×384 across 256 patches.


| k   | copy   | raw_dino | state  | TOTAL GAIN | LINEAR PROJ GAIN | TEMPORAL GAIN |
| --- | ------ | -------- | ------ | ---------- | ---------------- | ------------- |
| 1   | 1124.6 | 919.8    | 867.0  | 257.6      | 204.8            | 52.8          |
| 2   | 1525.0 | 1160.7   | 1100.2 | 424.8      | 364.3            | 60.5          |
| 3   | 1755.2 | 1279.8   | 1220.0 | 535.2      | 475.4            | 59.8          |
| 4   | 1875.0 | 1336.6   | 1279.8 | 595.2      | 538.4            | 56.8          |


- Same as CLS: state beats both copy and raw-DINO at every horizon.
- Temporal gain (~~53–60) slightly larger than CLS (~~38–51), but structural gain dominates in both cases.

**The punchline:**

Combined with the AR rollout results (model collapses past k=1 when you iterate `w_pred` on its own output): the information for multi-step prediction **is in the state** and **is linearly accessible**. The failure isn't "the state doesn't encode dynamics" — it's "the rollout mechanism (`w_pred` iterated on itself) can't extract that information." That's what motivates A1 (learn an explicit forward model `f(state) → next_state`) rather than A2 (learn multi-horizon decode heads, which would be redundant since linear decode already works).