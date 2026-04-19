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

## A2 (train with multi-horizon heads)

**HYPOTHESIS**:  multi-horizon prediction can be an impossible task for some cases (i.e for certain videos or for certain horizons). its possible sometimes that the video frames encoded so far may not always have information required to accurately predict over k=3 horizon.

**k-decay weighting**. So, maybe we should weigh each loss depending upon how far it is (weights for k = 1>2>3...)

**context for long-horizon prediction**. Also, maybe it is better to do long-horizon prediction, only if the model has encoded atleast 3 frames (can ablate over how many frames to encode before multi-horizon prediction kicks in), because to predict long-horizon the model needs to understand dynamics, and to understand dynamics it needs to ingest atleast two distinct frames

**VERIFYING HYPOTHESIS**: *state - dino_raw*  i.e stays ~5% at every horizon. If k=3,4 were systematically unlearnable, we'd expect the gap to collapse to ~0 at high k. It doesn't. This is a hint that the state carries comparable info for all k — arguing against k-decay weighting.

**NOTE**: right now, everything is terrible. once our predictor is good, then maybe can check this.

### To Do: verify this hypothesis via experiment



**ABLATIONS to run:**

1. one MLP + horizon embedding conditioning added to state

Inductive bias.

- separate MLP says: "horizons are totally separate tasks; each head does its own thing."
- this says: "horizons are the same task parameterized by k; predict-5-ahead should share structure with predict-4-ahead."

If DINO dynamics are smooth in k (the k=3 prediction should look like k=2's prediction moved slightly forward in time), B's smoothness prior is a win. If horizons really are different regimes (e.g., k=1 is local denoising, k=4 is scene-level prior), A lets each head specialize without gradient interference. 

## A2: K-ablation results — CLS

**Setup.** Train `w_pred_k` heads (separate MLP per k, k=1..K) with uniform 1/K weighting, all-t supervision, DINO-space, pred-only, 100 epochs. Identical config to `2ldiw9xk` (K=1 CLS baseline) aside from `encoder.rnn.max_horizon`. Runs: K=1 `2ldiw9xk` (baseline), K=2 `7zotsrwf`, K=3 `jk05gf14`, K=4 `hp9v42d1`. Patches runs still in flight.

**Evaluation 1: closed-form ridge probe on frozen `state_t`** (eval unchanged from B1 — this measures how *informative* the state is about x_{t+k}, regardless of the MLP head). Baselines are identical across K because they only depend on the SSv2 val features.

| k | copy   | raw-DINO | mean-pool | state K=1 (`2ldiw9xk`) | state K=2 (`7zotsrwf`) | state K=3 (`jk05gf14`) | state K=4 (`hp9v42d1`) |
|---|--------|----------|-----------|------------------------|------------------------|------------------------|------------------------|
| 1 |  632.3 |   562.1  |   649.1   |   524.0                |   525.7                |   529.4                |   533.5                |
| 2 |  925.3 |   779.6  |   788.8   |   730.2                |   726.4                |   725.8                |   726.6                |
| 3 | 1123.1 |   912.6  |   880.8   |   860.4                |   853.2                |   849.8                |   848.6                |
| 4 | 1238.3 |   986.8  |   936.3   |   936.0                |   927.8                |   923.3                |   921.1                |

**Evaluation 2: trained MLP heads** (wandb `eval/pred_loss_k{k}` at end of training — what the model actually learned to output).

| k | K=1 (`2ldiw9xk`) | K=2 (`7zotsrwf`) | K=3 (`jk05gf14`) | K=4 (`hp9v42d1`) |
|---|------------------|------------------|------------------|------------------|
| 1 |   513.0          |   515.6          |   521.0          |   525.6          |
| 2 |   —              |   720.4          |   720.7          |   722.6          |
| 3 |   —              |   —              |   852.7          |   852.1          |
| 4 |   —              |   —              |   —              |   935.1          |

**Takeaways.**

1. **Long-horizon state improves monotonically with K** (ridge probe). At k=4, state loss drops from 936.0 (K=1) → 921.1 (K=4), a 14.9 absolute improvement. At k=3, 860.4 → 848.6 (−11.8). Multi-horizon supervision shapes `state_t` to carry more k≥2 information — exactly the intended effect.

2. **Small k=1 penalty** (+9.5 from K=1 to K=4 on the ridge probe; +12.6 on the trained head). Uniform 1/K weighting dilutes the k=1 gradient by a factor of K, so k=1 gets less supervision in higher-K runs. Cheap to recover later with non-uniform weights if needed — not a dealbreaker.

3. **Trained heads beat the ridge probe at short horizons, saturate at long horizons.** k=1: trained head 513–526 vs. ridge 524–534 (trained wins by ~10). k=2: trained 720.4–722.6 vs. ridge 725.8–730.2 (trained wins by ~5). k=3: trained 852.1 vs. ridge 848.6 (ridge wins by ~4). k=4: trained 935.1 vs. ridge 921.1 (ridge wins by ~14). An MLP is strictly more expressive than a linear map, so ridge ≤ MLP with infinite data. At k=3,4 the ridge probe already matches or beats the trained head — suggesting the K=4 head is underfit at long horizons under uniform weighting + 100 epochs.

4. **State beats every baseline at every horizon for every K.** Even the weakest row (K=1 state at k=4: 936.0) essentially ties mean-pool (936.3) and clearly beats raw-DINO (986.8) and copy (1238.3). Monotone improvement over K strengthens this.

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

| k | copy   | raw-DINO | mean-pool | state K=1 (`e6esmgmu`) | state K=2 (`97x4ktzc`) | state K=3 (`0hymll1d`) | state K=4 (`zn1nvup2`) |
|---|--------|----------|-----------|------------------------|------------------------|------------------------|------------------------|
| 1 | 1124.6 |   919.7  |   993.1   |   867.0                |   867.4                |   872.9                |   877.4                |
| 2 | 1525.0 |  1160.7  |  1144.6   |  1100.2                |  1094.6                |  1094.8                |  1095.1                |
| 3 | 1755.2 |  1279.8  |  1229.1   |  1220.0                |  1212.1                |  1209.6                |  1207.8                |
| 4 | 1875.0 |  1336.6  |  1274.5   |  1279.8                |  1271.8                |  1268.6                |  1265.9                |

**Evaluation 2: trained MLP heads** (wandb `eval/pred_loss[_k{k}]`, most recent step).

| k | K=1 (`e6esmgmu`) | K=2 (`97x4ktzc`) | K=3 (`0hymll1d`) | K=4 (`zn1nvup2`) |
|---|------------------|------------------|------------------|------------------|
| 1 |    851.5 (final) |    858.1 (ep~24) |    862.7 (ep~24) |    865.8 (ep~24) |
| 2 |   —              |   1095.2         |   1096.9         |   1094.0         |
| 3 |   —              |   —              |   1222.3         |   1219.0         |
| 4 |   —              |   —              |   —              |   1288.1         |

**Takeaways (tentative, mid-training).**

1. **Same qualitative pattern as CLS, earlier in training.** Long-horizon state improves monotonically with K on the ridge probe: k=4 drops 1279.8 → 1265.9 (−13.9), k=3 drops 1220.0 → 1207.8 (−12.2). Small k=1 penalty (+10.4). Signal is already visible at epoch 24; expect it to strengthen by epoch 100.

2. **Mean-pool is a stronger baseline for patches** than for CLS. Mean-pool beats raw-DINO at k=2,3,4 (raw−mean gaps of −16, −51, −62). Per-token temporal averaging is informative for patches because most patches barely move (static background) — averaging is essentially a better prior. State still beats mean-pool at every horizon.

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


