### * Multi-horizon probe eval

- Given batch size = 32, T=8, dim = : the encoder spits outs (32, 8, D) - lets consider CLS for simplicity

**Step** **1**. **Run** **the** **encoder** **on** **all** **SSv2** **videos.** For each video (N=168k train, 24k val), run the RNN on all 8 frames. For each frame x_t, you get a state_t; a 384 dim vector.

**Step** **2**. **Collect** **(input,** **target)** **pairs** **for** **ridge** **regression.** For each horizon k (1 through 4), loop over valid time steps t=1 to T−k−1.

Same target, different inputs. For k=1 with T=8, that's 6 valid t values per video × 168k videos ≈ 1M training rows.

**Step** **3** **—** **Fit one** **linear** **probe per horizon** **(closed-form,** **on** **train).** For each k, solve:                                                                           

W* = argmin || ***input*** @ W − x_{t+k}||² + λ||W||²

Here input is taken from timestep t (can be *recurrent state* or *dino ground truth* feature), and used to predict the DINO feature at timestep=T+K

Solved in one shot via W = (XᵀX + λI)⁻¹ XᵀY — no SGD, no epochs, no hyperparameters to tune. The accumulators XᵀX and XᵀY are streamed in fp64, so this is numerically stable even over 1M rows.                                                            

**Step** **4** — **Evaluate** **on** **val.** Same loop over (video, t) pairs on the val split. Compute three errors per sample:


| Column    | Formula                        | What it measures                                                                                                                                                       |
| --------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| copy_last | L2(x_t, x_{t+k})               | **drift baseline:** "How much do frames drift over k steps?" A predictor that just outputs the current frame gets this error.                                          |
| last      | L2(x_t @ W_raw, x_{t+k})       | Best linear prediction from the current frame alone, no history. Captures any static/structural regularity (e.g. background persists, objects stay roughly in place). |
| state     | L2(state_t @ W_state, x_{t+k}) | Best linear prediction from the RNN's state after seeing frames 0..t. Has access to temporal history — could encode velocity, trend, periodic motion.                  |


Average each over all val (video, t) pairs → the numbers in the table.

**Derived columns (additive decomposition):**

Total improvement over copy_last decomposes cleanly into two parts:

```
(copy_last - state)  =  (copy_last - last)  +  (last - state)
 total improvement   structural gain       temporal gain
```


| Metric            | Formula             | What it measures                                                                                                                                                                 |
| ----------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| copy_last - state | `copy_last - state` | **Total improvement**: how much better the state probe is than copying last encoded frame                                                                                        |
| copy_last - last  | `copy_last - last`  | **Structural gain**: what a linear projection of the current frame buys you over just copying last encoded frame (static regularity, background persistence)                     |
| last - state      | `last - state`      | **Temporal gain**: what the encoder's temporal modeling adds on top on a linear projection of last frame.Positive means the state carries information beyond the current frame. |


## Can a linear regressor over last frame learn temporal dynamics?

- regression with L2 loss learns the conditional mean E[x_{t+k} | x_t] over the training set --> and this mean can encode actual temporal understanding through statistical regularity

##### What it can capture:

- **Appearance-conditioned future**: "features that look like a mid-swing arm tend to end up extended" — SSv2 data teaches this statistically, the probe picks it up. Not motion perception, but learned "what usually comes next given this appearance."
- **Systematic drift**: features shrink toward dataset mean as k grows (the mean-regression effect) — probe models it directly.
- **Object/scene identity persistence**: "same video → same background → future frame near current frame" — a large identity-like component of W captures this.

**What it cannot capture:**

- **Velocity / direction of motion**: you need at least two frames to observe change. A single frame tells you where things are, not where they're going. If two videos have the same x_t but opposite motion, the probe gives them the same prediction. Any predictor that distinguishes them needs x_{t-1} (or equivalent history).
- **Trajectory phase**: "object is 1/3 of the way through a throw" requires history to identify.
- **Ordering-dependent patterns**: by construction a permutation of the past doesn't affect x_t, so the probe is invariant to it.

So the linear regressor captures a lot of learned-from-dataset "temporal regularity" but zero motion-from-observation "temporal dynamics". These two things are often conflated under the word "temporal."

## Linear regressor over history

**What history gives you that one frame can't:**

- **Order-independent aggregation (denoising)**: averaging multiple DINO vectors of the same scene reduces per-frame noise and gives a better scene prior — bag-of-frames predictor
- **Order-dependent info (dynamics)**: velocity, acceleration, trajectory phase. Requires knowing *which frame came first* — a permutation-invariant predictor like **meanpool** cannot get this.

**The baselines, by what they can access:**


| baseline         | input to linear regressor             | captures                                                                       | misses                                             |
| ---------------- | ------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------- |
| `copy_last`      | none (identity: x = x_t); no learning | —                                                                              | everything                                         |
| `last`           | x_t                                   | appearance→future, mean-regression, scene identity                             | anything needing >1 frame (velocity, dynamics)     |
| `meanpool`       | mean(x_0..x_t)                        | + unordered aggregation / denoising                                            | ordering → no velocity, no trajectory phase        |
| `concat_history` | concat(x_0..x_t)                      | + linear dynamics from ordered aggregation (velocity, finite differences, EMA) | nonlinear dynamics (curvature, conditional motion) |
| `state`          | RNN state                             | + nonlinear dynamics (if training taught it)                                   | whatever training missed                           |


**Capacity**: `concat_history` can model both `meanpool` (learns to set each W = 1/T) and `last` (set each W =0 and last W=1) as a linear probe.

`meanpool` and `last` are **not nested**: aggregation helps at long k but hurts at k=1 (where x_t is near-optimal and averaging in older frames just adds noise — CLS numbers: last 562 vs meanpool 649 at k=1, flipping to 987 vs 936 at k=4). 

`state` sits off the linear chain — nonlinear encoding of the same frames `concat_history` sees.

**Full decomposition:**

```
copy_last − state  =  (copy_last − last)     LIN PROJ         (appearance, mean-regression)
                   +  (last − meanpool)     AGGREG           (can be negative at small k)
                   +  (meanpool − concat_history)  LIN DYNAMICS     (ordering, linear AR)
                   +  (concat_history − state)      NONLIN DYNAMICS  (the RNN's unique value)
```

The last column is the only one that specifically rewards the RNN for being nonlinear. If it's zero or negative, a linear AR model matches the RNN and the nonlinearity is wasted. `concat_history` is the baseline currently missing from Tables 1–3 — computing it would cleanly split "ordering helps a linear model" (LIN DYNAMICS) from "nonlinear processing helps beyond that" (NONLIN DYNAMICS).

### **Table 1: CLS Results** (wandbID `2ldiw9xk`, evalID `multi_horizon_probe`, 2026-04-19)


| k   | copy_last | last  | meanpool | concat_history | state | LIN PROJcopy_last− last | POOLINGlast−meanpool | ORDER HISTmeanpool−concat_history | NONLIN-DYNconcat_history−state |
| --- | --------- | ----- | -------- | -------------- | ----- | -------------------------- | ---------------------- | ------------------------------------ | -------------------------------- |
| 1   | 632.3     | 562.1 | 649.1    | 528.1          | 524.0 | 70.1                       | −87.0                  | 121.0                                | 4.1                              |
| 2   | 925.3     | 779.6 | 788.8    | 728.8          | 730.2 | 145.7                      | −9.2                   | 60.0                                 | −1.4                             |
| 3   | 1123.1    | 912.6 | 880.8    | 853.7          | 860.4 | 210.5                      | 31.9                   | 27.1                                 | −6.7                             |
| 4   | 1238.3    | 986.8 | 936.3    | 929.9          | 936.0 | 251.5                      | 50.5                   | 6.4                                  | −6.0                             |


### X-Y --> WHAT DOES THE Y BUY YOU?

1. **LINEAR PROJECTION OF X_T :**

- **perf. increases with k** (70 → 252): linear-regressor over last frame buys more at longer horizons
- x_t decorrelates from x_{t+k}, optimal linear predictor shrinks toward dataset mean.

1. **UNORDERED POOLING:**

- **k**: negative at k=1 (averaging older frames pollutes the signal; recency bias- x_T is near-optimal)
- positive at k=3,4 (x_t decorrelates, unordered averaging gets you the "scene" mean which is closer to distant future than x_t).

1. **ORDERED HISTORY**

- best performance at short horizon k=1/2; outperforms Lin. Proj. of last frame!!!
- **linear dynamics learnt from ordered history collapses with k** (121 → 60 → 27 → 6). Velocity / first-order-motion features fade fast with horizon.

1. **NON LINEAR DYNAMICS**

- Nonlinearity buys nothing beyond a linear combination of x_0..x_t.

#### OR --> RNN trained on next-frame prediction matches linear regressor trained for k=2, 3, 4 prediction!!!!!!!!!

### Possibilities:

- linear regressor over concat(x_0, ..., x_t) is near optimal for next-frame prediction (k=1,2)

### why is RNN state not better than a linear regressor over future horizon prediction?

Three compounding reasons:

1. **Training objective is k=1 only:**

- ****Multi-step info that ends up in `state_t` is incidental
- `lin_reg_history` is fit *directly* on the k of interest, with closed-form ridge — no objective mismatch.

1. **The 384dim state is a bottleneck**
2. **LayerNorm at every step destroys information** `lin_reg_history` **keeps.** `state ← LN(state + update)` re-normalizes magnitude at every step. Absolute feature scale — which the raw concat retains — is gone from the state. For long-horizon prediction (where mean-regression and scale matter), this is a real loss.

#### Feature scale and mean-regression are the same phenomenon viewed from two angles.

**Mean-regression as a scale operation.** 

The optimal long-k linear predictor is approximately `α(k)·x_t + (1 − α(k))·μ`, where μ is the dataset mean and α shrinks toward 0 as k grows. 

At k=1, α≈1 (predict ~x_t itself); at k=4, α≈0.4 (predict mostly the mean, slightly nudged by x_t). 

To execute this interpolation, the predictor needs `x_t` at its true magnitude — the contribution `α·x_t` only has the right scale if the input has the right scale. The probe is literally weighting "how much of x_t to keep" vs "how much to fall back to the mean," and that weighting is calibrated against the input's actual norm.

**What LN destroys.** `state ← LN(state + update)` re-normalizes to unit variance every step. After LN, every state vector has the same magnitude — only direction differs. So the probe can no longer distinguish "this is a high-energy frame, scale prediction up" from "this is a low-energy frame, scale prediction down." It outputs a fixed-magnitude prediction regardless. Two videos with the same direction but very different energies get the same forecast, even though their real `x_{t+k}` differ in magnitude.

**Why this hurts more at long horizons than short.** At k=1, scale matters less because the optimal prediction is dominated by direction (state ≈ next frame's direction). At k=4, the optimal prediction is mostly μ plus a small magnitude-aware correction — and that magnitude-aware correction is exactly what LN has wiped out. `lin_reg_history` keeps raw concat untouched, so it can apply the right α-shrink per video; the state can't.

So mean-regression requires preserved scale, and LayerNorm removes it. That's why the gap (`hist − state`) is small at k=1 (+4 in CLS) and grows to −6 at k=4 — the long-horizon "shrink toward mean" arithmetic only works for the predictor that still has magnitudes to shrink.

---

**Table 2: Patch Results** (wandbID `e6esmgmu`, evalID `multi_horizon_probe`, 2026-04-19) — token-wise probe; `state/last/meanpool` use shared `W: D×D = 384×384`, `concat_history` uses shared `W: T·D × D = 3072×384`, all fit across 256 spatial positions.


| k   | copy_last | last   | meanpool | concat_history | state  | LIN PROJcopy_last− last | POOLINGlast−meanpool | ORDER HISTmeanpool−concat_history | NONLIN-DYNconcat_history−state |
| --- | --------- | ------ | -------- | -------------- | ------ | ------------------------- | ---------------------- | ----------------------------------- | -------------------------------- |
| 1   | 1124.6    | 919.7  | 993.1    | 868.9          | 867.0  | 204.8                     | −73.4                  | 124.2                               | 2.0                              |
| 2   | 1525.0    | 1160.7 | 1144.6   | 1093.8         | 1100.2 | 364.3                     | 16.0                   | 50.8                                | −6.4                             |
| 3   | 1755.2    | 1279.8 | 1229.1   | 1208.5         | 1220.0 | 475.4                     | 50.7                   | 20.6                                | −11.6                            |
| 4   | 1875.0    | 1336.6 | 1274.5   | 1269.8         | 1279.8 | 538.4                     | 62.1                   | 4.7                                 | −10.0                            |


Pooling flips positive sooner for patches (k=2 vs k=3 for CLS). Patches are dominated by static background tokens, so unordered averaging becomes a better prior earlier. Consistent with the Table 6 static/dynamic split in meet1.

## TAKEAWAY SO FAR

**What Tables 1 and 2 actually establish (CLS + patches):**

- The RNN state is an 8× lossless linear-decoding substrate for multi-horizon prediction, obtained from k=1 supervision alone. It matches linear map trained with k=2,3,4  prediction with access to all previous frames.
- Next-frame prediction instills multi-horizon linear decodability at this horizon range.
- The short-lived first-order velocity signal from having ORDERED HISTORY and the long-horizon mean-regression from UNORDERED AGGREGATION are data properties of SSv2/DINO, not token-type artifacts.

**What NONLIN-DYN ≈ 0 does and does not say:**

- our RNN state matches linear probe, thus it gets all the "linear" temporal info. out of the data.

#### Q: Does the data (Dino features of SSV2 frames) have non-linear temporal structure?

Experiment:  MLP vs Linear probe on concat_history

##### Possibility 1: MLP = Linear Probe

- The predictable structure in the data is already linear.
- Our RNN state reaches the Linear AR ceiling --> does *lossless compression* + *multi-horizon transfer from k=1*

#### Possibility 2: MLP > Linear Probe

- There's nonlinear structure in the raw frames
- We need to do better -  different encoder (transformer, deeper RNN, different update rule, different loss).

Future:

- The sharp follow-up is to compute concat_history on A2 runs: if A2 state drops below concat_history at k≥2, multi-horizon supervision pushed the state past the linear-AR ceiling.

### MLP probe Tier 1 — CLS result (2026-04-19)

Ran on `2ldiw9xk` (evalID `mlp_probe`). Design: residual MLP (linear skip + zero-init GELU nonlinear branch). Skip warm-started to the closed-form ridge solution and **frozen** during SGD — so the nonlinear branch only needs to find residual corrections on top of ridge. Guarantees final val ≤ ridge (nonlinear can stay at 0), so "MLP ≈ ridge" cannot be confused with under-fit.

 **Δhist > 0** means the raw 8-frame DINO features, by themselves, contain nonlinear temporal structure that a linear map can't extract

**Table 3 — CLS (**`2ldiw9xk`**)**


| k   | ridge(state) | mlp(state) | Δstate | ridge(hist) | mlp(hist) | Δhist | Δhist %   |
| --- | ------------ | ---------- | ------ | ----------- | --------- | ----- | --------- |
| 1   | 524.0        | 520.2      | 3.8    | 528.1       | 524.9     | 3.2   | **0.61%** |
| 2   | 730.2        | 723.5      | 6.7    | 728.8       | 724.1     | 4.7   | **0.65%** |
| 3   | 860.4        | 852.3      | 8.1    | 853.7       | 848.1     | 5.6   | **0.66%** |
| 4   | 936.0        | 928.6      | 7.4    | 929.9       | 924.3     | 5.6   | **0.60%** |


**Δhist ~0.6% at every k → Possibility 1 for CLS.** The nonlinear function class can only shave ~0.6% off the linear-AR ceiling. Within the ~1% noise band specified in the decision rule.

Δstate ≈ Δhist across k: whatever small nonlinear signal the MLP finds in the raw concat is also accessible from the RNN state. Consistent with "state is a lossless substrate + data has negligible extractable nonlinear structure."

### MLP probe Tier 1 — Patches result (2026-04-20)

Ran on `e6esmgmu` (same design and hyperparameters as CLS). Per-token probe with one shared MLP across 256 spatial positions.

**Table 4 — Patches (`e6esmgmu`)**


| k   | ridge(state) | mlp(state) | Δstate | ridge(hist) | mlp(hist) | Δhist | Δhist %   |
| --- | ------------ | ---------- | ------ | ----------- | --------- | ----- | --------- |
| 1   | 867.0        | 848.7      | 18.3   | 868.9       | 847.3     | 21.6  | **2.49%** |
| 2   | 1100.2       | 1075.6     | 24.6   | 1093.8      | 1072.3    | 21.5  | **1.96%** |
| 3   | 1220.0       | 1192.3     | 27.7   | 1208.5      | 1187.3    | 21.2  | **1.75%** |
| 4   | 1279.8       | 1252.3     | 27.5   | 1269.8      | 1248.2    | 21.6  | **1.70%** |


Linear probe gets most of the info out; MLP probe achieves ~1-2% improvement - it could just be better static-token predictor

### Combined CLS + Patches reading


|                      | CLS `2ldiw9xk`         | Patches `e6esmgmu`     |
| -------------------- | ---------------------- | ---------------------- |
| Δhist range across k | 0.60–0.66%             | 1.70–2.49%             |
| Decision             | Possibility 1          | Possibility 2 (weak)   |
| Token aggregation    | single semantic vector | 256 per-spatial tokens |


The two results are consistent with an encoder-side story, not a data-side one: **the same raw DINO frames have extractable nonlinear structure that survives in patches but is linearly-summarized away in the CLS token.**

**A1 / A2 implications:**

- The linear-AR-ish ceiling was pinned within ~1% on CLS but can be exceeded by ~2% on patches with a nonlinear probe.
- A2 K=4 CLS state sits within 1% of all ceilings — A2 training buys something but bounded tightly.
- On patches, A2 K=4 state (still training; current ridge-probe snapshot at k=4 = 1265.9) is 1.4% ABOVE the MLP-on-hist ceiling of 1248.2 — still headroom. Re-running MLP probe on the converged A2 K=4 patches checkpoint (`zn1nvup2` `last.pt` after epoch 100) will tell us whether multi-horizon supervision on patches can cross the MLP-on-concat ceiling.

**Tier 2 / 3 decision.** Patches Δhist at 1.7–2.5% is marginal relative to our MLP capacity. A Tier 3 capacity ablation (3 MLP sizes, k=4 on patches) would tighten the true nonlinear ceiling; Tier 2 (MLP on `last` / `meanpool`) would localize where the nonlinear gain lives (single-frame vs unordered aggregation vs ordered history). Both worthwhile for patches; skip for CLS (clear Poss 1).

Full numbers and hyperparameters: `docs/exp_progress.md` → 2026-04-19 / 2026-04-20 MLP probe entries.

## Are we at the upper-bound of multi-horizon prediction performance for SSV2 DINO?

**Short answer:** for CLS, yes — A2 is already at the ceiling. For patches, maybe 1-2% headroom, but probably not worth chasing.

**CLS math (k=4):**

- K=1 state: 936.0
- Linear-AR ceiling (ridge on concat_history): 929.9
- MLP ceiling (on concat_history): 924.3
- A2 K=4 state: **921.1** ← already at/below MLP ceiling
- Total headroom above K=1 ≈ 1.3%; A2 K=4 already captured ~1.6%. Further K sweeps, loss weighting, horizon embeddings = chasing <0.5% sub-noise gains. **Not worth the compute.**

**Patches math (k=4):**

- K=1 state: 1279.8
- MLP ceiling (on concat_history): 1248.2
- A2 K=4 state (ep ~24): 1265.9 ← still 1.4% above MLP ceiling
- Two problems with banking on this headroom:
  1. The 1.7% MLP-over-ridge gap itself is suspect — likely static-token appearance modeling, not temporal dynamics.

**Bigger picture:** We're near the upper bound of what SSv2 + frozen DINO can support for k=1..4 prediction. `copy_last → state` already **captures the bulk (~25% L2 reduction);** 

### Real gains require changing the substrate, not refining A2:

- Features with temporal awareness (DINO is single-frame).
- Task with richer dynamics than SSv2's short clips (*NEED NON-LINEARITY DATA IN ORDER TO MODEL IT*)
- Architectures that can exploit genuine nonlinearity

**Immediate next step before burning more A2 cycles:** run Tier 2 `MLP(last)` on patches. If Δ(MLP(last) − ridge(last)) ≈ Δhist, the patches ceiling is a static-appearance artifact and A2 has no real headroom — pivot off A2.

---

#### doesn't make sense now, but went into multi-horizon training rabbit hole, confounding variables to ablate, training decisions... will help later :)

A2 (train with multi-horizon heads) section moved to `docs/A2_multi_horizon_training.md`.

---

### Where do we stand:  SSv2-8f with DINO is saturated

### Potential directions

- Need new multi-horizon prediction benchmark?
  - need proxy eval for stage 2, since i.e the point of stage 1
- how good are we at next-frame prediction (patches)
  - sigreg
  - dinoWM
- Recurrent update - more expressive update, TTT
  - explore if we lag behind causal transformer
- Modelling multimodal futures

## Next hill to climb?

right now, we want to improve stage 1 encoder. To do that, we need benchmarks that show where/what the RNN encoder lacks. 

### Three different tangents to improve stage 1 encoder:

1. Evaluate RNN compression for stage 2 (build stage 2 eval; maybe compression is already good enough).
2. Improve stage 1 encoder (T=64, TTT, capacity, aux losses).
3. Close RNN-vs-transformer gap on patch prediction.

There are two things we care about **spatio-temporal compression** in stage 1:

- should capture first-order dynamics; needed to predict nearby future frames / model scene-flow
- should capture semantics; needed for long-horizon forecasting

# 1. Stage 2 proxy benchmark

**Stage-2 proxy benchmark = same multi-horizon probe, one hierarchy up:**


| axis    | stage 1 (done)    | stage 2 (new)                                                          |
| ------- | ----------------- | ---------------------------------------------------------------------- |
| unit    | frame             | chunk (8 frames → 1 state)                                             |
| k       | 1..4 frames       | 1..8 chunks (~4s each)                                                 |
| horizon | ≤3s               | ~30s–few min                                                           |
| data    | SSv2 (2-6s clips) | Epic-Kitchens-100 or COIN (multi-stage, 10-20min)                      |
| metric  | per-frame L2      | per-chunk L2 + **ratio: stage2(states) / stage2(uncompressed concat)** |


- multiple decision - what dataset, metric, unit. 
- lets make task progressively harder -->  we stick to our existing evaluation setup i.e frame prediction evaluated with per-frame L2 but increasing the context length. 
- Currently, our RNN state matches linear regressor over concat(x_0, ...,x_t) baseline at SSV2 8 frames. ***what about 64 frames?***   
- **BASELINE**: lets get a baseline that really outperfroms our RNN state so that hillclimbing that metric will lead to better RNN state

### Increasing horizon vs context

- as horizon increases, cov(x_t, x_{t+k}) reduces, and prediction --> conditional mean
- to do really long horizon prediction, you need to operate at higher temporal abstraction.
- recurrence trained with next few-frame prediciton can minimize spatio-temporal redundancy. its not enough to operate at higher temporal abstraction. for that, we will have to train autoregressive transformer that operates a chunk levels. 
- to get to that, lets first get as much juice as possible from the recurrent state - long context, as lossless compression, meaningful compression that enables great long horizon chunk-level prediction (for stage 2).

## Evaluating long-horizon dynamics

### **the stage-2 setup itself measures how good RNN compression is**++ ; requires operating at higher abstraction

#### Short horizon (k=1..4 frames, ~1s):

- next frame ≈ current frame + small motion. 
- Local velocity/finite-differences dominate the prediction.

**GOAL**: *Can we get a compressed spatio-temporal chunk that preserves semantic information necessary for long horizon prediction?*

#### Long horizon (chunk k=4, ~16s ahead)

- motion decorrelates completely. 
- What actually predicts chunk_{t+4} is "*what activity is this*" and "*what stage of the activity are we in*"
- **semantic/categorical info, not kinematic**. 
- A k=1-trained encoder has no pressure to encode such priors; a long-horizon or contrastive objective does.

### Chunk-level forecasting --> semantics / higher abstraction

1. Take long videos (Epic-Kitchens / COIN).
  1. Slice into 8-frame chunks.
  2. Stage-1 RNN on each chunk → one 384-dim state per chunk.
  3. Train a causal transformer over the chunk-state sequence to predict future chunk features.
  4. Baseline: same transformer, but tokens are raw chunk-DINO (3072-dim per chunk).
  5. Gap between them = compression loss. Climb it by improving stage-1 (state dim, arch, aux losses, K).
    **No frame-level gymnastics needed — the stage-2 setup itself measures how good RNN compression is**++

*The test asks*: whatever the RNN throws away during that 8→1 compression actually needed for predicting future chunks? 

- If concat-baseline beats state-baseline, yes — something useful got discarded. 
- Same question as stage 1, just evaluated at the scale where stage 2 will use it.

#### Q: Is the goal to improve stage-1 encoders, or to qualify stage 1 for stage 2? These are separate metrics.\

# 2. Improve stage 1 encoder (T=64, TTT, capacity, aux losses)

**GOAL**: *Can we get a compressed spatio-temporal chunk that preserves semantic information necessary for long horizon prediction?*

### Widening ***state-vs-baseline gaps.***

#### Long-horizon prediction (K=3,4) may require more context

- right now we evaluate, **All-t, no warmup.** multi_horizon_[probe.py](http://probe.py) evaluates anchors t=1..T−k−1. For qk=4, T=8 → t ∈ {1,2,3}: **sometimes the state has seen only 2 frames before predicting 4 ahead**
- those low-t large-k samples contribute to the average, likely dragging it down. Gating to t ≥ k (or t ≥ 3) would be a cleaner eval and ***might slightly widen state-vs-baseline gaps.***

#### train on longer videos.

- Training on 32-64 frame sequences at the same 384-dim state forces harder compression; hard to outperform W(concat) baseline.
- the alternative (increasing horizon i.e longer k) is dominated by mean-regression as I argued earlier.

### Recurrent update - more expressive update, TTT

- explore if we lag behind causal transformer

### Encoder capacity

- RNN: wider `w_pred` MLP (`pred_hidden_dim` config), deeper MLP, bigger state dim, or cross-patch attention in the update.

### Auxiliary / alternative training objectives

- A k=1-trained encoder has no pressure to encode such priors; a long-horizon or contrastive objective does.

### Next-frame prediction quality (patches)

- sigreg
- dinoWM

# 3. Close RNN-causal transformer gap on patches

outperforming causal transformer baseline for patch tokens (table 3 in @docs/[meet1.md](http://meet1.md)) can be one goal, need to find out how much of the improvement is from static tokens compared to dynamic tokens - need to add causal transformer column in table 6 in meet1.md?

#### Q: doin static vs patch decomposition on multi-horizon probe? - a lossy way to find how much of the win is temporal?

### RNN vs Causal Transformer — decoder and encoder expressivity

**Decoder answer: no, not matched.**

- **RNN decoder (`w_pred`):** 2-layer MLP, `Linear(384→384) → ReLU → Linear(384→384)`. ~295K params. Defined in `root/models/rnn.py:132-143, 167`.
- **Causal transformer decoder (`pred_head`):** single `Linear(384→384)` + a LayerNorm. ~148K params. Defined in `root/models/causal_transformer.py:28-29, 70`.

So the transformer has a *less* expressive decoder and still wins on patches — if anything this strengthens the "transformer encoder is carrying more info" reading, not weakens it. But it means any "trained head" cross-arch comparison (e.g., Table 3 trained-head column) is not apples-to-apples on the decoder axis. The ridge probe sidesteps this entirely — that's one more reason it's the cleaner comparison.

**Encoder expressivity: not matched — the transformer has ~15× more parameters.**

RNN update machinery (`GatedTransformerCore`, surprise mode):

- `w_precision`: 384×384 linear (~148K)
- `w_pred`: 2-layer MLP (~295K)
- `encoder`: `Identity` in DINO-space mode
- One LN
- Total ≈ 440K trainable params, applied recurrently T=8 times with a single 384-D state.

Causal transformer (`CausalTransformerPredictor`, defaults `depth=4, n_heads=8, d_ff=4*dim`):

- 4 × `TransformerEncoderLayer`: each ≈ 4×(384²) attention + 2×(384×1536) FFN ≈ 1.77M params
- Frame/patch embeddings, `pred_head`, LN: negligible
- Total ≈ 7.1M trainable params, single forward pass over 8 frames with full attention over 2048 patch tokens.

So the 62-L2 patches gap (851 vs 789) is confounded: 15× params, full cross-token attention, and bigger "state" (the full token sequence vs a single 384-D vector) all differ simultaneously. "The RNN encoder is weaker" is one explanation; "the RNN has much less capacity" is another, and currently we can't distinguish them.

If you want a fair comparison, you'd scale up either:

- RNN: wider `w_pred` MLP (`pred_hidden_dim` config), deeper MLP, bigger state dim, or cross-patch attention in the update.
- Transformer: fewer layers / smaller d_ff to match ~500K params.

