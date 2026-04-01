# March 31 Conversation

## Baseline Comparison (predict_in_dino_space run, ~100k steps)

| Baseline | L2 |
|---|---|
| Predict global mean | 1820 |
| Copy current frame (no learning) | 620 |
| **Your model** | **517** |

Your model at 517 is only ~17% better than the **copy-current-frame baseline** (620). That's not great — it means `w_pred(state)` is barely doing more than echoing the current input.

But it's not "bad" either — it's not collapsed (unlike the previous run). The model is learning *something*, just not much beyond the trivial "next frame ≈ current frame" prior, which is strong in SSv2 at frame_step=4.

**Should you make W_pred more expressive?** The current `w_pred` is a 2-layer MLP (384→384→384). That's reasonably expressive for this dimensionality. The bottleneck is more likely that:

1. The **state itself** doesn't contain enough temporal context for prediction — it's updated via `state + W_prec(error)` which is quite simple
2. The prediction problem is genuinely hard: DINO features change unpredictably between frames (note the high variance: std=607)

---

## What are the baselines?

- **Predict global mean (1820)**: If `w_pred` just output the average DINO feature across all videos and all frames — the dumbest possible predictor. Your model is way better than this.

- **Copy current frame (620)**: If `w_pred` perfectly copied the current frame's DINO feature as its prediction for the next frame — i.e. "nothing changes." This is the real baseline to beat, since consecutive video frames are visually similar.

---

## What is the L2 lower bound?

Zero would mean perfectly predicting the next DINO feature, but that's unrealistic. The practical lower bound is the **irreducible noise** — the inherently unpredictable component of frame-to-frame variation.

Hard to know without an oracle. But a rough proxy: train a powerful model (e.g. large transformer) with access to all previous frames to predict the next, and see where it saturates.

The high std (607) relative to the mean (620) in consecutive frame L2 suggests there's huge variance across samples — some clips barely change (L2 ≈ 0), others change drastically. So the "true" lower bound is probably not near zero for the dataset as a whole.

---

## On W_enc, collapse, and general representations

### Context
When predicting `h = w_enc(dino)` without action classification, model collapsed. But with action classification, pred loss reduced to (train=0.3, val=3.6) — clearly not collapsed. But how do we know the model was actually predicting dynamics/future? What subspace does W_enc map to? Why do we need W_enc?

### Why W_enc exists (the JEPA argument)

Raw DINO space contains everything — including stuff that's *inherently unpredictable* (exact textures, lighting, irrelevant background). The L2 lower bound is high because you're penalized for things no model could predict. W_enc's role is to learn a subspace that **discards the unpredictable and retains only what's forecastable**. Prediction becomes easier and more meaningful in this space.

The problem: "discard unpredictable stuff" and "discard everything" look the same to the pred loss. Hence collapse.

### How JEPA/world models prevent collapse without action classification

- **V-JEPA**: EMA target encoder. W_enc_target is a slow-moving copy of W_enc, not trained by gradient. It can't co-adapt with the predictor to collapse — the predictor must actually learn dynamics against a moving but stable target.
- **VICReg**: Explicit regularization — maintain variance across the batch, decorrelate dimensions. Directly prevents the representation from becoming degenerate.
- **Contrastive**: Don't predict the exact vector, just make the prediction *closer* to the true next frame than to other frames. Collapse gives chance-level contrast, so it's penalized.

### The CE+pred run (train=0.3, val=3.6)

Can't be sure the model was predicting real dynamics vs. a shortcut. The CE loss forces W_enc to preserve action-discriminative information, so the subspace is non-trivial. But it's *biased* — W_enc only preserves what helps classify 174 SSv2 actions. Two videos with identical actions but different object trajectories could map to the same W_enc output. The pred loss in this space only measures "can I predict the next frame's action-relevant features," not general dynamics. The train/val gap (0.3 vs 3.6) also suggests overfitting in this learned space.

### What if you want a general video representation?

This is the world model objective: the state should be a **sufficient statistic for predicting all future observations**. Here's the tension:

- **Predict in DINO space** (predict_in_dino_space=True): No collapse possible, forces the state to capture rich visual dynamics. But high irreducible error, and the model wastes capacity on unpredictable aspects.
- **Predict in learned space** (W_enc): Can focus on what's predictable and meaningful. But needs a collapse prevention mechanism.

The current predict_in_dino_space setup is actually underrated. The state (384-dim) IS the compressed representation — it accumulates info via error-driven updates, and `w_pred` must map from this compressed state back to full DINO space. The state is free to learn its own internal structure; only the *prediction target* is fixed. The bottleneck is whether `w_pred` and the update rule are expressive enough, not whether the target space is right.

**If you want the best of both worlds** (learned subspace + no collapse), the simplest option is **EMA target encoder**: add a copy of W_enc that's updated via exponential moving average, use it to compute the prediction target. This is a few lines of code and is exactly what V-JEPA does.
