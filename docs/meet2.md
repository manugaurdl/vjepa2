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

