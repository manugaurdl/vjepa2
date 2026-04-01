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

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Evaluating Whether Recurrent State Has General-Purpose Visual Features

### Context
I am doing next frame dino prediction - whether I predict dino or predict h = W(Dino) with some trick to prevent collapse doesn't matter right now - there is a correct algorithm solution that enables my model to predict future, capture dynamics and store it in a spatio-temporal state; to find what works and doesn't I need a good eval setup to find if my recurrent state is has good, general purpose visual features; for example when i train with action classifcation + predict H, my model is not collapsed. but i am certain the recurrent state is primed for features that can be used to predict action or primed for ssv2 dataset- after training is completed i need a way to evaluate this. lets discuss what are good ways to do this

### Eval Hierarchy

#### Tier 1: Transfer probes (do features generalize beyond SSv2?)

**Linear probe on a different dataset with different label space.** This is the most direct test. If your state is just encoding SSv2 action prototypes, it'll fail on tasks requiring different visual concepts.

- **Kinetics-400 linear probe**: Different action taxonomy, different visual distribution. If SSv2-trained features transfer well here, they're not SSv2-specific.
- **Diving48 linear probe**: Requires fine-grained temporal discrimination (dive types differ only in motion sequence). Tests if the state captures motion dynamics, not just object/scene identity.
- **UCF101 or HMDB51**: Classic benchmarks, easy to set up, give you a calibration point against published numbers.

The key: freeze the encoder entirely, only train a linear head. If you need a nonlinear head to get decent accuracy, the features aren't linearly accessible — they're entangled.

#### Tier 2: Multi-horizon prediction decay (Eval A from stage1.md)

Already defined and excellent. But the critical addition: **run it on out-of-distribution clips**. If you train on SSv2, evaluate the decay curve on Epic-Kitchens or Kinetics clips. A state that truly captures dynamics should predict future DINO features reasonably well even on video from a different distribution. If the decay curve is flat on SSv2 but steep on Kinetics, your state has memorized SSv2-specific temporal patterns.

#### Tier 3: Probes for specific properties the state should/shouldn't encode

**Should encode (dynamics):**
- Temporal ordering: given two states S_a, S_b from the same video, can a linear probe tell which came first? A state that captures dynamics should make this easy.
- Speed/motion magnitude: use optical flow magnitude as pseudo-GT, probe whether it's linearly decodable from the state.

**Should NOT encode (appearance-only):**
- Scene classification: if your state is equally good at "kitchen vs outdoor vs gym" as a single-frame DINO feature, it's storing appearance, not dynamics. The state *should* be somewhat worse at pure scene classification than raw DINO (it traded appearance bits for dynamics bits).

This asymmetry is the diagnostic: **good dynamics features should be better than single-frame DINO on temporal tasks and comparable-or-worse on static tasks.**

#### Tier 4: Nearest-neighbor qualitative analysis

Embed a bunch of clips from mixed datasets, find nearest neighbors in state space. If neighbors share dynamics (hand moving left, object falling) across visually different scenes — that's a general representation. If neighbors are just visually similar frames from SSv2 — it's overfit.

#### What to do first

Start with **one transfer linear probe** (Kinetics-400 or UCF101) and **multi-horizon decay on out-of-distribution video**. These two alone will tell you:

1. Is the state useful beyond SSv2? (transfer probe)
2. Does the state capture predictable dynamics or just dataset-specific patterns? (OOD decay curve)

Everything else is for diagnosing *why* something fails once you know *that* it fails.
