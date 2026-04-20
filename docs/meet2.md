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

**Table 3 — CLS (**`2ldiw9xk`**)**


| k   | ridge(state) | mlp(state) | Δstate | ridge(hist) | mlp(hist) | Δhist | Δhist %   |
| --- | ------------ | ---------- | ------ | ----------- | --------- | ----- | --------- |
| 1   | 524.0        | 520.2      | 3.8    | 528.1       | 524.9     | 3.2   | **0.61%** |
| 2   | 730.2        | 723.5      | 6.7    | 728.8       | 724.1     | 4.7   | **0.65%** |
| 3   | 860.4        | 852.3      | 8.1    | 853.7       | 848.1     | 5.6   | **0.66%** |
| 4   | 936.0        | 928.6      | 7.4    | 929.9       | 924.3     | 5.6   | **0.60%** |


**Δhist ~0.6% at every k → Possibility 1 for CLS.** The nonlinear function class can only shave ~0.6% off the linear-AR ceiling. Within the ~1% noise band specified in the decision rule.

Δstate ≈ Δhist across k: whatever small nonlinear signal the MLP finds in the raw concat is also accessible from the RNN state. Consistent with "state is a lossless substrate + data has negligible extractable nonlinear structure."

Patches result pending. If patches agrees with CLS, the combined reading is: on this data + encoder, linear AR is the ceiling (not a probe-class artifact). The 8× lossless compression + multi-horizon transfer from k=1 supervision is as strong a result as can be expected without changing the data or the encoder. A1 remains motivated by the need for a rollout mechanism over the compact state; A2's headroom is bounded by the linear-AR ceiling itself, which is now pinned.

Full numbers and hyperparameters: `docs/exp_progress.md` → 2026-04-19 MLP probe entry.

---

## A2 (train with multi-horizon heads)

**HYPOTHESIS**:  multi-horizon prediction can be an impossible task for some cases (i.e for certain videos or for certain horizons). its possible sometimes that the video frames encoded so far may not always have information required to accurately predict over k=3 horizon.

**k-decay weighting**. So, maybe we should weigh each loss depending upon how far it is (weights for k = 1>2>3...)

**context for long-horizon prediction**. Also, maybe it is better to do long-horizon prediction, only if the model has encoded atleast 3 frames (can ablate over how many frames to encode before multi-horizon prediction kicks in), because to predict long-horizon the model needs to understand dynamics, and to understand dynamics it needs to ingest atleast two distinct frames

**VERIFYING HYPOTHESIS**: *state - last*  i.e stays ~5% at every horizon. If k=3,4 were systematically unlearnable, we'd expect the gap to collapse to ~0 at high k. It doesn't. This is a hint that the state carries comparable info for all k — arguing against k-decay weighting.

**NOTE**: right now, everything is terrible. once our predictor is good, then maybe can check this.

### To Do: verify this hypothesis via experiment

**ABLATIONS to run:**

1. one MLP + horizon embedding conditioning added to state

Inductive bias.

- separate MLP says: "horizons are totally separate tasks; each head does its own thing."
- this says: "horizons are the same task parameterized by k; predict-5-ahead should share structure with predict-4-ahead."

If DINO dynamics are smooth in k (the k=3 prediction should look like k=2's prediction moved slightly forward in time), B's smoothness prior is a win. If horizons really are different regimes (e.g., k=1 is local denoising, k=4 is scene-level prior), A lets each head specialize without gradient interference. 

1. supervision:
  1. all -t : predict all future frames.

## A2: K-ablation results — CLS

**Setup.** Train `w_pred_k` heads (separate MLP per k, k=1..K) with uniform 1/K weighting, all-t supervision, DINO-space, pred-only, 100 epochs. Identical config to `2ldiw9xk` (K=1 CLS baseline) aside from `encoder.rnn.max_horizon`. Runs: K=1 `2ldiw9xk` (baseline), K=2 `7zotsrwf`, K=3 `jk05gf14`, K=4 `hp9v42d1`. Patches runs still in flight.

**Evaluation 1: closed-form ridge probe on frozen `state_t`** (eval unchanged from B1 — this measures how *informative* the state is about x_{t+k}, regardless of the MLP head). Baselines are identical across K because they only depend on the SSv2 val features.

**Table 4**


| k   | copy_last | last  | meanpool | state K=1 (`2ldiw9xk`) | state K=2 (`7zotsrwf`) | state K=3 (`jk05gf14`) | state K=4 (`hp9v42d1`) |
| --- | --------- | ----- | -------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| 1   | 632.3     | 562.1 | 649.1    | 524.0                  | 525.7                  | 529.4                  | 533.5                  |
| 2   | 925.3     | 779.6 | 788.8    | 730.2                  | 726.4                  | 725.8                  | 726.6                  |
| 3   | 1123.1    | 912.6 | 880.8    | 860.4                  | 853.2                  | 849.8                  | 848.6                  |
| 4   | 1238.3    | 986.8 | 936.3    | 936.0                  | 927.8                  | 923.3                  | 921.1                  |


when linear probing, 

- meanpooling frames until T beats a linear regressor of DINO (for k>=2)
- for a given k ; do they train only with a single K head  or upto K heads

**Evaluation 2: trained MLP heads** (wandb `eval/pred_loss_k{k}` at end of training — what the model actually learned to output).

**Table 4**


| k   | K=1 (`2ldiw9xk`) | K=2 (`7zotsrwf`) | K=3 (`jk05gf14`) | K=4 (`hp9v42d1`) |
| --- | ---------------- | ---------------- | ---------------- | ---------------- |
| 1   | 513.0            | 515.6            | 521.0            | 525.6            |
| 2   | —                | 720.4            | 720.7            | 722.6            |
| 3   | —                | —                | 852.7            | 852.1            |
| 4   | —                | —                | —                | 935.1            |


**Takeaways.**

1. **Long-horizon state improves monotonically with K** (ridge probe). At k=4, state loss drops from 936.0 (K=1) → 921.1 (K=4), a 14.9 absolute improvement. At k=3, 860.4 → 848.6 (−11.8). Multi-horizon supervision shapes `state_t` to carry more k≥2 information — exactly the intended effect.
2. **Small k=1 penalty** (+9.5 from K=1 to K=4 on the ridge probe; +12.6 on the trained head). Uniform 1/K weighting dilutes the k=1 gradient by a factor of K, so k=1 gets less supervision in higher-K runs. Cheap to recover later with non-uniform weights if needed — not a dealbreaker.
3. **Trained heads beat the ridge probe at short horizons, saturate at long horizons.** k=1: trained head 513–526 vs. ridge 524–534 (trained wins by ~10). k=2: trained 720.4–722.6 vs. ridge 725.8–730.2 (trained wins by ~5). k=3: trained 852.1 vs. ridge 848.6 (ridge wins by ~4). k=4: trained 935.1 vs. ridge 921.1 (ridge wins by ~14). An MLP is strictly more expressive than a linear map, so ridge ≤ MLP with infinite data. At k=3,4 the ridge probe already matches or beats the trained head — suggesting the K=4 head is underfit at long horizons under uniform weighting + 100 epochs.
4. **State beats every baseline at every horizon for every K.** Even the weakest row (K=1 state at k=4: 936.0) essentially ties meanpool (936.3) and clearly beats last (986.8) and copy_last (1238.3). Monotone improvement over K strengthens this.

**Punchline vs. B1.** B1's framing was that the failure was in the rollout mechanism, not the state, and A2 would be "redundant." A2 partially contradicts B1: the state *can* be pushed further — the K=4 training run improves long-horizon state informativeness by ~1.5% (k=4) without any rollout changes. The trained heads are a cheap, working alternative to iterated `w_pred` for k≤2; at k≥3 they're at parity with the closed-form probe, suggesting diminishing returns from MLP expressivity alone under this training regime. A1 (explicit forward model) is still the right next step for genuine rollout, but A2 is not a dead end — the state moved.

**Reproduce (CLS).**

```bash
# Train (per-K, on one GPU; 100 epochs pred-only)
python train.py --config base --wandb.run_name pred_in_dino_space_mh_k<K> \
    --load_cache_feats --no-action_classification \
    --encoder.rnn.predict_in_dino_space --encoder.rnn.max_horizon <K> --epochs 100

# Evaluate (ridge probe on frozen state)
PYTHONPATH=/home/manu/vjepa2 python root/evals/multi_horizon_probe.py \
    --checkpoint /nas/manu/vjepa2/outputs/pred_in_dino_space_mh_k<K>_<wandbID>/last.pt \
    --data_dir /nas/manu --max_horizon 4 --batch_size 128 --gpu 0

# Pull trained-head numbers
.venv/bin/python scripts/pull_mh_wandb_cls.py
```

## A2: K-ablation results — Patches (mid-training snapshot)

**Caveat.** Patches K∈{2,3,4} runs (`97x4ktzc`, `0hymll1d`, `zn1nvup2`) are still training — snapshot at **epoch ~24/100** (copied from `last.pt` to `/nas/manu/vjepa2/outputs/snapshots_2026-04-18/`). K=1 reference (`e6esmgmu`) is fully trained (100 epochs). Not apples-to-apples; expect the K>1 numbers to improve further at convergence.

**Evaluation 1: closed-form ridge probe on frozen `state_t`.**


| k   | copy_last | last   | meanpool | state K=1 (`e6esmgmu`) | state K=2 (`97x4ktzc`) | state K=3 (`0hymll1d`) | state K=4 (`zn1nvup2`) |
| --- | --------- | ------ | -------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| 1   | 1124.6    | 919.7  | 993.1    | 867.0                  | 867.4                  | 872.9                  | 877.4                  |
| 2   | 1525.0    | 1160.7 | 1144.6   | 1100.2                 | 1094.6                 | 1094.8                 | 1095.1                 |
| 3   | 1755.2    | 1279.8 | 1229.1   | 1220.0                 | 1212.1                 | 1209.6                 | 1207.8                 |
| 4   | 1875.0    | 1336.6 | 1274.5   | 1279.8                 | 1271.8                 | 1268.6                 | 1265.9                 |


**Evaluation 2: trained MLP heads** (wandb `eval/pred_loss[_k{k}]`, most recent step).


| k   | K=1 (`e6esmgmu`) | K=2 (`97x4ktzc`) | K=3 (`0hymll1d`) | K=4 (`zn1nvup2`) |
| --- | ---------------- | ---------------- | ---------------- | ---------------- |
| 1   | 851.5 (final)    | 858.1 (ep~24)    | 862.7 (ep~24)    | 865.8 (ep~24)    |
| 2   | —                | 1095.2           | 1096.9           | 1094.0           |
| 3   | —                | —                | 1222.3           | 1219.0           |
| 4   | —                | —                | —                | 1288.1           |


**Takeaways (tentative, mid-training).**

1. **Same qualitative pattern as CLS, earlier in training.** Long-horizon state improves monotonically with K on the ridge probe: k=4 drops 1279.8 → 1265.9 (−13.9), k=3 drops 1220.0 → 1207.8 (−12.2). Small k=1 penalty (+10.4). Signal is already visible at epoch 24; expect it to strengthen by epoch 100.
2. **Meanpool is a stronger baseline for patches** than for CLS. Meanpool beats last at k=2,3,4 (raw−mean gaps of −16, −51, −62). Per-token temporal averaging is informative for patches because most patches barely move (static background) — averaging is essentially a better prior. State still beats meanpool at every horizon.
3. **Trained patch heads at k=1 degrade slightly with K** (851.5 → 865.8) — same 1/K-dilution effect as CLS. At k≥2 they're tight across K (within ~3).
4. **Ridge probe still below trained head at k=3,4** just like CLS: ridge 1207.8 vs. trained 1219.0 (k=3); ridge 1265.9 vs. trained 1288.1 (k=4). MLP is underfit at long horizons — consistent with CLS finding that 1/K weighting + 100 epochs isn't enough head capacity at long k.

**Next.** Re-run probe eval on the patches `last.pt` after the 3 runs finish training (epoch 100) to get the converged comparison. Expect gaps to widen — K=4 should pull further below K=1 at k=3,4.

**Reproduce (patches).**

```bash
# Train (per-K; patches use batch_size 8 due to S=256)
python train.py --config base --wandb.run_name patch_pred_dino_space_mh_k<K> \
    --load_cache_feats --use_patch_tokens --no-action_classification \
    --encoder.rnn.predict_in_dino_space --encoder.rnn.max_horizon <K> \
    --batch_size 8 --val_batch_size 8 --epochs 100

# Evaluate (ridge probe on frozen state)
PYTHONPATH=/home/manu/vjepa2 python root/evals/multi_horizon_probe.py \
    --checkpoint /nas/manu/vjepa2/outputs/patch_pred_dino_space_mh_k<K>_<wandbID>/last.pt \
    --data_dir /nas/manu --max_horizon 4 --batch_size 32 --gpu 0

# Pull trained-head numbers
.venv/bin/python scripts/pull_mh_wandb_patches.py
```

