# Experiment Progress

## What We Care About
The primary goal is building a good **next-frame predictor** in DINO feature space, then analyzing **memory dynamics** to understand how the RNN accumulates temporal information. Classification accuracy is secondary — useful as a sanity check but not the main objective.

### Key metrics (priority order):
1. **eval/pred_loss & eval/pred_error_l2**: Is the model learning to predict next frames? Per-timestep plot reveals if prediction improves or collapses.
2. **eval/h_t_norm**: Should increase over time as novel info accumulates (e.g., 15→17 over 8 frames). Flat = state not accumulating = likely collapse.
3. **eval/update_norm**: Healthy: large at t=0, drops at t=1, then slightly increases as novel info appears. Drops to ~0 after t=0 = trivial solution.
4. **eval/cos_sim**: Direction similarity between consecutive hidden states. Should show meaningful variation. Flat = state direction frozen.
5. **eval/r_novelty**: Ratio of novel info in updates. Should be non-trivial across timesteps.
6. **eval/memory_l2**: L2 shift between consecutive states. Non-zero = meaningful temporal dynamics.
7. **eval/acc**: Only relevant when action_classification=True.

### Collapse detection checklist:
- pred_loss drops to near-zero suspiciously fast
- h_t_norm flat across timesteps
- update_norm → 0 after t=0
- cos_sim flat
- All together = representation collapse. Root cause: if prediction target is in a learned space (W_enc @ dino_feat), the encoder can collapse to make prediction trivial. Fix: predict in frozen DINO space (predict_in_dino_space=True).

## Eval Metrics Reference
- **eval/acc**: top-1 classification accuracy on SSv2 validation set (only meaningful when action_classification=True)
- **eval/ce_loss**: cross-entropy loss on validation set
- **eval/pred_loss**: L2 next-frame prediction error (mean over timesteps 1+, skipping t=0)
- **eval/total_loss**: ce_loss + pred_loss_weight * pred_loss (or just pred_loss when action_classification=False)
- **eval/pred_error_l2**: plotly line plot — per-timestep L2 prediction error across the clip
- **eval/update_gate**: plotly line plot — per-timestep gating values
- **eval/update_norm**: plotly line plot — per-timestep L2 norm of the state update vector
- **eval/r_novelty**: plotly line plot — ratio of novel information in the update
- **eval/memory_l2**: plotly line plot — L2 shift between consecutive hidden states
- **eval/cos_sim**: plotly line plot — cosine similarity between consecutive hidden states
- **eval/h_t_norm**: plotly line plot — norm of hidden state over time

## Train Metrics
- **trainer/loss, trainer/total_loss**: running average total loss
- **trainer/ce_loss**: running average CE loss
- **trainer/pred_loss**: running average weighted pred loss
- **trainer/lr**: current learning rate
- **trainer/iter_ms_avg**: average iteration time in ms

---

## Experiments

### 1. `zyvsy8gk` — RNN, CE + pred, CLS, learned space

- **Run name**: `update=w(error)_L2weight1e-1`
- **Checkpoint**: `/nas/manu/vjepa2/outputs/update=w(error)_L2weight1e-1_zyvsy8gk/best.pt`
- **Config**: `action_classification=True`, `next_frame_pred=True`, `pred_loss_weight=0.1`, `update_type=surprise`, `use_patch_tokens=False`, `predict_in_dino_space=False`, `epochs=100`
- **Train command**:
```bash
python train.py --config base --wandb.run_name update=w(error)_L2weight1e-1 \
    --data_dir /nas/manu --output_dir /nas/manu/vjepa2/outputs \
    --load_cache_feats --action_classification --next_frame_pred \
    --pred_loss_weight 0.1 --epochs 100
```
- **Results**: Table 1 (UCF101 84.0%), Table 4 (normal=2.23, shuffled=2.29, ratio=1.03x)

**Reproduce Table 1 row:**
```bash
PYTHONPATH=/home/manu/vjepa2 python eval_transfer.py \
    --checkpoint /nas/manu/vjepa2/outputs/update=w(error)_L2weight1e-1_zyvsy8gk/best.pt \
    --mode probe --train_csv /nas/manu/ucf101/data/train.csv \
    --data_csv /nas/manu/ucf101/data/test.csv \
    --num_classes 101 --cache_features --no_aug
```

**Reproduce Table 4 row:**
```bash
PYTHONPATH=/home/manu/vjepa2 python root/evals/temporal_shuffle_test.py \
    --checkpoint /nas/manu/vjepa2/outputs/update=w(error)_L2weight1e-1_zyvsy8gk/best.pt \
    --data_dir /nas/manu --batch_size 64 --num_shuffles 3 --gpu 0
```

---

### 2. `2ldiw9xk` — RNN, pred only, CLS, DINO space

- **Run name**: `pred_only_dino_space` (or similar)
- **Checkpoint**: `/nas/manu/vjepa2/outputs/<run_name>_2ldiw9xk/best.pt`
- **Config**: `action_classification=False`, `predict_in_dino_space=True`, `use_patch_tokens=False`, `epochs=100`
- **Train command**:
```bash
python train.py --config base --wandb.run_name pred_only_dino_space \
    --data_dir /nas/manu --output_dir /nas/manu/vjepa2/outputs \
    --load_cache_feats --no-action_classification \
    --encoder.rnn.predict_in_dino_space --epochs 100
```
- **Results**: Table 1 (UCF101 85.4%), Table 2 (RNN=176.9 — note: the 513 in Table 2 is stale, from ~100k steps), Table 4 (normal=176.9, shuffled=211.9, ratio=1.20x, copy=620)

**Reproduce Table 2 RNN row / Table 4 row:**
```bash
PYTHONPATH=/home/manu/vjepa2 python root/evals/temporal_shuffle_test.py \
    --checkpoint /nas/manu/vjepa2/outputs/<run_name>_2ldiw9xk/best.pt \
    --data_dir /nas/manu --batch_size 64 --num_shuffles 3 --gpu 0
```

**Reproduce Table 1 row:**
```bash
PYTHONPATH=/home/manu/vjepa2 python eval_transfer.py \
    --checkpoint /nas/manu/vjepa2/outputs/<run_name>_2ldiw9xk/best.pt \
    --mode probe --train_csv /nas/manu/ucf101/data/train.csv \
    --data_csv /nas/manu/ucf101/data/test.csv \
    --num_classes 101 --cache_features --no_aug
```

---

### 3. `e6esmgmu` — RNN, pred only, patches, DINO space

- **Run name**: `patch_pred_dino_space`
- **Checkpoint**: `/nas/manu/vjepa2/outputs/patch_pred_dino_space_e6esmgmu/last.pt`
- **Config**: `action_classification=False`, `predict_in_dino_space=True`, `use_patch_tokens=True`, `epochs=100`
- **Train command**:
```bash
python train.py --config base --wandb.run_name patch_pred_dino_space \
    --data_dir /nas/manu --output_dir /nas/manu/vjepa2/outputs \
    --load_cache_feats --use_patch_tokens --no-action_classification \
    --encoder.rnn.predict_in_dino_space --batch_size 8 --val_batch_size 8 --epochs 100
```
- **Results**: Table 1 (UCF101 81.7%), Table 3 (RNN=851), Table 4 (normal=851.5, shuffled=1114.1, ratio=1.31x, copy=1116), static/dynamic decomp (dynamic 24.7%, static 16.2%)

**Reproduce Table 3 RNN row / Table 4 row:**
```bash
PYTHONPATH=/home/manu/vjepa2 python root/evals/temporal_shuffle_test.py \
    --checkpoint /nas/manu/vjepa2/outputs/patch_pred_dino_space_e6esmgmu/last.pt \
    --data_dir /nas/manu --batch_size 64 --num_shuffles 3 --gpu 0
```

**Reproduce static/dynamic decomposition:**
```bash
PYTHONPATH=/home/manu/vjepa2 python root/evals/static_dynamic_decomposition.py \
    --checkpoint /nas/manu/vjepa2/outputs/patch_pred_dino_space_e6esmgmu/last.pt \
    --data_dir /nas/manu --batch_size 64 --gpu 0
```

**Reproduce Table 1 row:**
```bash
PYTHONPATH=/home/manu/vjepa2 python eval_transfer.py \
    --checkpoint /nas/manu/vjepa2/outputs/patch_pred_dino_space_e6esmgmu/last.pt \
    --mode probe --train_csv /nas/manu/ucf101/data/train.csv \
    --data_csv /nas/manu/ucf101/data/test.csv \
    --num_classes 101 --cache_features --no_aug
```

---

### 4. `tj9x820q` — RNN, CE + pred, patches, learned space

- **Run name**: `patch_ce_pred`
- **Checkpoint**: `/nas/manu/vjepa2/outputs/patch_ce_pred_tj9x820q/best.pt`
- **Config**: `action_classification=True`, `predict_in_dino_space=False`, `use_patch_tokens=True`, `pred_loss_weight=0.1`, `epochs=100`
- **Train command**:
```bash
python train.py --config base --wandb.run_name patch_ce_pred \
    --data_dir /nas/manu --output_dir /nas/manu/vjepa2/outputs \
    --load_cache_feats --use_patch_tokens --action_classification \
    --pred_loss_weight 0.1 --batch_size 8 --val_batch_size 8 --epochs 100
```
- **Results**: Table 1 (UCF101 78.3%), Table 4 (normal=3.80, shuffled=4.29, ratio=1.13x, copy=—)

**Reproduce Table 4 row:**
```bash
PYTHONPATH=/home/manu/vjepa2 python root/evals/temporal_shuffle_test.py \
    --checkpoint /nas/manu/vjepa2/outputs/patch_ce_pred_tj9x820q/best.pt \
    --data_dir /nas/manu --batch_size 64 --num_shuffles 3 --gpu 0
```

---

### 5. `ud2ncxlq` — Causal Transformer, pred only, CLS, DINO space

- **Checkpoint**: checkpoint path unknown — check wandb
- **Config**: `encoder.type=causal_transformer`, `action_classification=False`, `use_patch_tokens=False`, evaluated at ~27 epochs
- **Train command**:
```bash
python train.py --config base --wandb.run_name causal_pred_cls \
    --data_dir /nas/manu --output_dir /nas/manu/vjepa2/outputs \
    --load_cache_feats --no-action_classification \
    --encoder.type causal_transformer --epochs 100
```
- **Results**: Table 2 (Causal Transformer=517, at ~27 epochs)
- **Note**: 517 is from an early checkpoint (~27 epochs). Fully trained number unknown.

---

### 6. `r55x2lcn` — Causal Transformer, pred only, patches, DINO space

- **Checkpoint**: checkpoint path unknown — check wandb
- **Config**: `encoder.type=causal_transformer`, `action_classification=False`, `use_patch_tokens=True`
- **Train command**:
```bash
python train.py --config base --wandb.run_name causal_pred_patches \
    --data_dir /nas/manu --output_dir /nas/manu/vjepa2/outputs \
    --load_cache_feats --use_patch_tokens --no-action_classification \
    --encoder.type causal_transformer --batch_size 8 --val_batch_size 8 --epochs 100
```
- **Results**: Table 3 (Causal Transformer=783)

---

## Tools
- **analyze_wandb_run.py**: `python analyze_wandb_run.py <user>/<project>/<run_id>` — fetches and prints final eval/train metrics. Use `wandb.Api(timeout=60)`.

## Transfer & OOD Evaluation

### Eval 1: Transfer Linear Probe (UCF101)

```bash
PYTHONPATH=/home/manu/vjepa2 python eval_transfer.py \
    --checkpoint /path/to/best.pt --mode probe \
    --train_csv /nas/manu/ucf101/data/train.csv \
    --data_csv /nas/manu/ucf101/data/test.csv \
    --num_classes 101 --cache_features --no_aug

# DINO mean-pool baseline (run once, reuse)
PYTHONPATH=/home/manu/vjepa2 python eval_transfer.py \
    --checkpoint /path/to/best.pt --mode probe --baseline \
    --train_csv /nas/manu/ucf101/data/train.csv \
    --data_csv /nas/manu/ucf101/data/test.csv \
    --num_classes 101 --cache_features --no_aug
```

### Eval 2: OOD Prediction Error Decay Curve

```bash
PYTHONPATH=/home/manu/vjepa2 python eval_transfer.py \
    --checkpoint /path/to/best.pt --mode decay \
    --data_csv /nas/manu/ucf101/data/test.csv \
    --ssv2_val_csv /nas/manu/ssv2/data/validation.csv \
    --output_dir outputs/eval_decay
```

## Reproduction Log

### 2026-04-08 — Tables 4 & 5 (temporal shuffle + copy shuffle sensitivity) reproduced

Reproduced with the padding-fixed `temporal_shuffle_test.py` (truncates CLS cache to 24777). Patch rows match published values exactly; CLS rows shifted upward because the padding bug was previously biasing ratios toward 1 (zero-padded rows gave ≈equal loss under normal and shuffled, adding a constant to both num and denom).

**Table 4 (model temporal shuffle test, SSv2 val, 3 seeds).** Absolute pred_loss is in incompatible units across rows: DINO-space rows are raw DINO L2 (comparable to Tables 2/3 and to copy baseline); learned-space rows are in a W_enc-projected space and only the ratio is interpretable.

| Model | Normal | Shuffled | Ratio (repro) | Published ratio |
|---|---|---|---|---|
| RNN CLS, CE+pred, learned space (`zyvsy8gk`)   | 3.70   | 4.13    | **1.12x** | 1.03x |
| RNN CLS, pred only, DINO space (`2ldiw9xk`)    | 512.96 | 752.44  | **1.47x** | 1.20x |
| RNN Patches, pred only, DINO space (`e6esmgmu`)| 851.46 | 1113.76 | **1.31x** | 1.31x |
| RNN Patches, CE+pred, learned space (`tj9x820q`)| 3.59 | 4.08    | **1.14x** | 1.13x |

**Table 5 (data-only copy baseline shuffle sensitivity, 3 seeds).** Within-video permutation: for each video, shuffle its 8 frames and recompute consecutive-pair L2.

| Row | Normal | Shuffled | Copy ratio (repro) | Published |
|---|---|---|---|---|
| Patches (DINO) | 1088.71 | 1587.54 | **1.46x** | 1.46x |
| CLS (DINO)     | 609.13  | 1003.72 | **1.65x** | 11.2x |

Patches exactly match; CLS does not. Checked: within-video gives 1.65x, cross-video (random frame from any other video) gives 5.95x. Neither reproduces 11.2x. The patches row reproduces exactly at 1.46x using the within-video method, so the methodology is right — the published 11.2x for CLS is likely an earlier buggy number; the narrative "random CLS pairs are very different" is only true cross-video, and even then the ratio is ~6x not 11x.

**Reinterpretation (important — this rewrites the Table 4 CLS story):**

Old story for `2ldiw9xk` (CLS DINO): shuffled model (211.9) still much better than copy (620), so the CLS model "learned both multi-frame aggregation (order-independent, survives shuffling) and some dynamics (20% degradation from shuffling)."

New story: shuffled model (**752.44**) is **worse than copy (609.13)**. The CLS model's entire edge over copy is order-dependent — just like patches. When frames are shuffled, the model is strictly worse than a copy-last-frame baseline. The "multi-frame aggregation survives shuffling" finding was an artifact of the padding bug; the correct picture is that both CLS and patch RNNs use temporal order to beat copy, and neither has an order-independent aggregation advantage.

**Caveat on the shuffle test itself:** ratio > 1 only proves the model is not permutation-invariant. It does NOT distinguish "learned trajectories" from "smart EMA / recency bias" — an EMA also degrades under shuffling because the most-recent frame is weighted highest. The right experiment for "did it learn dynamics?" is autoregressive rollout (freeze state at t, predict t+1..t+k without GT, compare error growth to copy(t→t+k)). Still to do.

Reproduce: `bash scripts/repro/table4_temporal_shuffle.sh` and `bash scripts/repro/table5_copy_shuffle_ratio.sh`.

### 2026-04-08 — Table 3 (SSv2 next-frame pred_loss, PATCHES) reproduced

Same metric as Table 2 but in DINO patch-token space (S=256, D=384). Reads `validation_patches.pt`, which is correctly sized `(24777, 8, 256, 384)` — built by `precompute_patch_feats.py`, not by `train.py --cache_dino_feats`, so the padding bug from Table 2 does not apply here. All three rows match published values within ~1% (small drift likely from fp16→fp32 cast path / chunking, published values were rounded).

| Model | Checkpoint | pred_loss (repro) | Published |
|---|---|---|---|
| Copy-last-frame baseline                                       | (data only)                                | **1088.71** | 1085 |
| Causal Transformer, pred only, patches DINO (`r55x2lcn`)       | `causal_pred_patches_r55x2lcn/last.pt`     | **788.60**  | 783  |
| RNN, pred only, patches DINO (`e6esmgmu`)                      | `patch_pred_dino_space_e6esmgmu/last.pt`   | **851.46**  | 851  |

Reproduce: `bash scripts/repro/table3_ssv2_pred_loss_patches.sh`. Copy baseline is computed in chunks of 256 videos because the full diff tensor is ~60 GB. Both model rows use `last.pt` (no `best.pt` exists — pred-only runs predate the best.pt fix in `train.py:349`).

**Reinterpretation:** Causal Transformer (789) clearly beats RNN (851), unlike CLS where they were equal (517 vs 513). For patches the state IS a bottleneck — explicit attention over all 8×256=2048 past patch tokens lets the transformer model patch-level motion that the RNN's single state can't track. Both still beat copy (1089), so both learn *something* beyond frame smoothness.

### 2026-04-08 — Table 2 (SSv2 next-frame pred_loss, CLS) reproduced + corrected

**Bug found and fixed.** Previously published numbers (Causal Transformer 100.1, RNN 176.9) were computed by averaging `pred_error_l2` over the full `validation.pt` file, which is zero-padded to 168913 rows. Only the first 24777 rows are real validation samples (see `validation.csv`). The remaining ~144k zero rows dilute the mean by ~7×. Root cause: `root/models/model.py:74` allocated `id_to_feat` with the train-split size unconditionally; train.py then saved the whole buffer. Fixed by trimming at save time and by truncating in every reader (`temporal_shuffle_test.py`, repro scripts). `validation_patches.pt` is unaffected — `precompute_patch_feats.py` sized it correctly.

**Corrected Table 2** (mean over N=24777 valid val samples, t=1..7, CLS DINO space):

| Model | Checkpoint | pred_loss (corrected) | Old (padded) |
|---|---|---|---|
| Copy-last-frame baseline                                | (data only)                                | **609.13** | 609 (was already truncated) |
| Causal Transformer, pred only, CLS DINO (`ud2ncxlq`)    | `causal_pred_cls_ud2ncxlq/last.pt`         | **517.06** | 100.1 |
| RNN, pred only, CLS DINO (`2ldiw9xk`)                   | `pred_in_dino_space_2ldiw9xk/last.pt`      | **512.96** | 176.9 |

**Reinterpretation:** RNN (513) ≈ Causal Transformer (517) ≈ ~16% better than copy (609). The qualitative finding "single-state RNN matches a transformer with full memory for CLS, so the state isn't a bottleneck" still holds, but the absolute headroom over copy is much smaller than the old numbers suggested. The earlier "Transformer at 100.1" upper-bound claim is invalid — `wandb`-logged val pred_loss from `train.py` was always correct (the val dataset iterates only real indices); the bug only hit standalone evals that loaded `validation.pt` directly.

Reproduce: `bash scripts/repro/table2_ssv2_pred_loss_cls.sh`. Both Table 2 model rows currently use `last.pt` because the runs predate the best-checkpoint fix described below — no `best.pt` exists on disk for either. Any newly trained pred-only run will save `best.pt` correctly.

**Follow-up fixes (2026-04-08):**
- `root/models/model.py:74` — `id_to_feat` is no longer a magic `zeros(168913, ...)`. It now sizes itself from `args.train_dataset_len`/`args.val_dataset_len` (set by `train.py:241-243` from the actual datasets) and raises if those aren't set. The save sites in `train.py` already trim with `model.id_to_feat[:len(loader.dataset)]`, so any newly cached `validation.pt` is correctly sized.
- `train.py:349` — `best.pt` save criterion now branches on `action_classification`: highest val acc when classifying, lowest val pred_loss when pred-only. Previously pred-only runs (e.g. `2ldiw9xk`, `e6esmgmu`) never wrote a `best.pt` because `acc > best_acc` was always false (acc stayed 0). Tracks `best_pred_loss` separately, seeded from the initial val.

### 2026-04-08 — Table 1 (UCF101 linear probe) reproduced

Re-ran all 6 rows of `docs/april_3_meet.md` Table 1 in parallel (one GPU each), 20 epochs, AdamW lr=1e-3, batch=64, no aug, cached features. All match within ≤0.5pt of the meeting doc.

| Model | Checkpoint | best_acc (repro) | Table 1 |
|---|---|---|---|
| DINO mean-pool                          | (baseline, no ckpt)                              | **88.03%** | 88.0% |
| DINO concat                             | (baseline, no ckpt)                              | **86.10%** | 86.0% |
| RNN CLS, DINO-space pred (`2ldiw9xk`)   | `pred_in_dino_space_2ldiw9xk/last.pt`            | **85.38%** | 85.4% |
| RNN CLS, CE+pred learned (`zyvsy8gk`)   | `update=w(error)_L2weight1e-1_zyvsy8gk/best.pt`  | **84.01%** | 84.0% |
| RNN Patch, DINO-space pred (`e6esmgmu`) | `patch_pred_dino_space_e6esmgmu/last.pt`         | **81.71%** | 81.7% |
| RNN Patch, CE+pred learned (`tj9x820q`) | `patch_ce_pred_tj9x820q/best.pt`                 | **78.83%** | 78.3% |

Reproduction artifacts in `outputs/repro_table1/`. Reproduce with `bash scripts/repro/table1_ucf101_probe.sh` (see `docs/repo_context.md` → UCF101 Linear Probe).

### 2026-04-08 — Autoregressive rollout (evalID: `autoregressive_rollout`) first results

New eval: build RNN state from first `t_ctx=4` frames, then iterate `w_pred` K=4 times (`w_pred(state) → w_pred(w_pred(state)) → ...`) with no new input. Per-horizon L2 vs copy/linear baselines. Eval + rationale documented in `docs/repo_context.md` → "Autoregressive Rollout" and `docs/april_3_meet.md` → rollout section.

**`2ldiw9xk` — RNN CLS, pred only, DINO space** (`pred_in_dino_space_2ldiw9xk/last.pt`)

| horizon k | copy   | linear   | model   | model/copy |
|-----------|--------|----------|---------|------------|
| 1         | 743.4  | 1894.6   | **606.6** | 0.82x ← extrap |
| 2         | 992.6  | 4922.9   | 3303.9  | 3.33x ← collapse |
| 3         | 1134.3 | 9360.0   | 3964.1  | 3.50x ← collapse |
| 4         | 1204.4 | 15124.0  | 4742.3  | 3.94x ← collapse |

**`e6esmgmu` — RNN Patches, pred only, DINO space** (`patch_pred_dino_space_e6esmgmu/last.pt`)

| horizon k | copy   | linear   | model    | model/copy |
|-----------|--------|----------|----------|------------|
| 1         | 1270.9 | 3380.2   | **957.8**  | 0.75x ← extrap |
| 2         | 1604.4 | 8622.2   | 4487.3   | 2.80x ← collapse |
| 3         | 1767.5 | 16243.2  | 7685.6   | 4.35x ← collapse |
| 4         | 1855.6 | 26219.1  | 15682.8  | 8.45x ← collapse |

**Interpretation.** Both models beat copy at horizon 1 — unsurprising, that's literally the training objective (at t_ctx=4 the model has seen real frames 0..3, and `w_pred(state_3)` is trained to predict frame 4). But from k=2 onward, iterating `w_pred` on its own output diverges rapidly from the DINO manifold, and the model is strictly worse than a constant-last-frame predictor. Neither model has a stable attractor under iterated self-application of `w_pred`.

This is consistent with the "smart 1-step predictor, not a dynamics model" hypothesis. It does NOT yet rule out extrapolation in an absolute sense — the way the architecture is trained, `w_pred` expects a real state as input, not its own output. A more faithful "AR rollout" would feed each prediction back as `h` into the surprise update, but that update has `error = pred - w_pred(state) = 0` and is degenerate (see `docs/april_3_meet.md` rollout section for the derivation). The clean conclusion: **under the only nondegenerate rollout available for this architecture, both DINO-space RNNs collapse past horizon 1.** Patches collapse faster than CLS in absolute ratio (8.4x vs 3.9x at k=4), consistent with patches living on a higher-dim manifold where iterated `w_pred` has more room to wander off.

Linear baseline is useless here: DINO features have large per-frame noise, so `x_t + k·(x_t - x_{t-1})` blows up way above even the collapsed model. Keep it in the table anyway as a "dumb dynamics" sanity check.

Reproduce:
```bash
# CLS
PYTHONPATH=/home/manu/vjepa2 python root/evals/autoregressive_rollout.py \
    --checkpoint /nas/manu/vjepa2/outputs/pred_in_dino_space_2ldiw9xk/last.pt \
    --data_dir /nas/manu --t_ctx 4 --batch_size 256 --gpu 0
# Patches
PYTHONPATH=/home/manu/vjepa2 python root/evals/autoregressive_rollout.py \
    --checkpoint /nas/manu/vjepa2/outputs/patch_pred_dino_space_e6esmgmu/last.pt \
    --data_dir /nas/manu --t_ctx 4 --batch_size 32 --gpu 0
```

### 2026-04-09 — Multi-horizon linear probe (evalID: `multi_horizon_probe`) first results

New eval: closed-form ridge regression `state_t @ W_k ≈ x_{t+k}` for k=1..4 on SSv2, fit in parallel with raw-DINO probe `x_t @ W_k_raw ≈ x_{t+k}`. The gap is exactly "what the encoder contributes to horizon k on top of linearly projecting the current frame forward." Eval documented in `docs/repo_context.md` → "Multi-Horizon Linear Probe." Motivation: the AR rollout eval (2026-04-08) collapsed past k=1, but that eval is architecturally degenerate for this surprise-update RNN (feeding prediction back gives `error=0`). B1 sidesteps the rollout mechanics and probes the state directly.

**`2ldiw9xk` — RNN CLS, pred only, DINO space** (`pred_in_dino_space_2ldiw9xk/last.pt`)

| k | copy    | raw-DINO | state   | raw−state gap | state/copy | state/raw |
|---|---------|----------|---------|---------------|-----------|-----------|
| 1 |  632.25 |  562.14  |  524.04 |  38.10        | 0.829x    | 0.932x    |
| 2 |  925.29 |  779.61  |  730.24 |  49.37        | 0.789x    | 0.937x    |
| 3 | 1123.09 |  912.61  |  860.37 |  52.25        | 0.766x    | 0.943x    |
| 4 | 1238.34 |  986.79  |  935.96 |  50.84        | 0.756x    | 0.948x    |

State beats both copy and raw-DINO probe at every horizon. The `raw − state` gap stays ~40–55 across k — the state consistently carries multi-step information beyond what's linearly in the current frame. Small gap in absolute terms, but consistent and in the right direction.

**Interpretation.** Combined with the AR rollout collapse: the **state is fine**, the rollout *mechanism* is the bottleneck. The information needed for 2–4 step prediction is present in `state_t` and linearly decodable; the problem is that iterating `w_pred` on its own output doesn't trace a trajectory through that state space. This argues for A1 (explicit forward/transition model `f(state) → next_state`, DreamerV3-style) over A2 (multi-horizon heads) — you don't need to re-learn the information, you need a rollout operator that actually exists.

**`e6esmgmu` — RNN Patches, pred only, DINO space** (`patch_pred_dino_space_e6esmgmu/last.pt`)

Token-wise probe: shared `W: 384×384` fit across all 256 spatial positions.

| k | copy    | raw-DINO | state   | raw−state gap | state/copy | state/raw |
|---|---------|----------|---------|---------------|-----------|-----------|
| 1 | 1124.58 |  919.75  |  866.98 |  52.77        | 0.771x    | 0.943x    |
| 2 | 1524.97 | 1160.65  | 1100.17 |  60.49        | 0.721x    | 0.948x    |
| 3 | 1755.22 | 1279.83  | 1220.04 |  59.78        | 0.695x    | 0.953x    |
| 4 | 1875.01 | 1336.61  | 1279.79 |  56.83        | 0.683x    | 0.957x    |

Same shape as CLS: state beats both copy and raw-DINO at every horizon, with `raw − state` gap ~50–60. The patch model encodes a little more per-token information beyond current frame than CLS does (in absolute L2), but the *relative* gap (`state/raw` ~0.95) is nearly identical to CLS. The per-patch temporal signal is real but modest — consistent with the static_dynamic result that most of the improvement lives in a small dynamic-patch minority.

**Joint takeaway (CLS + patches).** Both models pass B1: the state is a strict improvement over raw DINO for multi-step prediction. Combined with the AR rollout collapse at k≥2, this pins the failure mode on the rollout mechanism, not on the state's information content. Clear motivation for A1 (explicit forward/transition operator) ahead of A2 (multi-horizon decode heads).

Reproduce:
```bash
# CLS
PYTHONPATH=/home/manu/vjepa2 python root/evals/multi_horizon_probe.py \
    --checkpoint /nas/manu/vjepa2/outputs/pred_in_dino_space_2ldiw9xk/last.pt \
    --data_dir /nas/manu --max_horizon 4 --batch_size 128 --gpu 0
# Patches
PYTHONPATH=/home/manu/vjepa2 python root/evals/multi_horizon_probe.py \
    --checkpoint /nas/manu/vjepa2/outputs/patch_pred_dino_space_e6esmgmu/last.pt \
    --data_dir /nas/manu --max_horizon 4 --batch_size 32 --gpu 0
```

### 2026-04-18 — A2 multi-horizon training, CLS K-ablation (`7zotsrwf`, `jk05gf14`, `hp9v42d1`)

Three new training runs: CLS, pred-only, DINO-space, uniform 1/K horizon weighting, all-t supervision, 100 epochs — identical config to `2ldiw9xk` baseline aside from `encoder.rnn.max_horizon`. Each adds K−1 extra MLP heads (`w_pred_extra`) for predicting x_{t+k} from `state_t` at k=2..K; the original `w_pred` (k=1 head) is unchanged and still drives the surprise update. Implementation: commit `97452db`, verified via 5 checks (shape, K=1 numerical invariant `mh[0] == pred_error_l2[:, 1:]`, backward flow, strict=True checkpoint load of `2ldiw9xk`/`e6esmgmu`, `multi_horizon_probe.py` on `2ldiw9xk` reproduces Table 7 exactly).

**Runs.**

| K | wandbID       | run_name                         | checkpoint                                                                   |
|---|---------------|----------------------------------|------------------------------------------------------------------------------|
| 2 | `7zotsrwf`    | `pred_in_dino_space_mh_k2`       | `/nas/manu/vjepa2/outputs/pred_in_dino_space_mh_k2_7zotsrwf/last.pt`         |
| 3 | `jk05gf14`    | `pred_in_dino_space_mh_k3`       | `/nas/manu/vjepa2/outputs/pred_in_dino_space_mh_k3_jk05gf14/last.pt`         |
| 4 | `hp9v42d1`    | `pred_in_dino_space_mh_k4`       | `/nas/manu/vjepa2/outputs/pred_in_dino_space_mh_k4_hp9v42d1/last.pt`         |

**Train command (template).**
```bash
python train.py --config base --wandb.run_name pred_in_dino_space_mh_k<K> \
    --load_cache_feats --no-action_classification \
    --encoder.rnn.predict_in_dino_space --encoder.rnn.max_horizon <K> --epochs 100
```

**Multi-horizon probe (ridge on frozen `state_t`).**

| k | `2ldiw9xk` K=1 | `7zotsrwf` K=2 | `jk05gf14` K=3 | `hp9v42d1` K=4 |
|---|----------------|----------------|----------------|----------------|
| 1 |   524.0        |   525.7        |   529.4        |   533.5        |
| 2 |   730.2        |   726.4        |   725.8        |   726.6        |
| 3 |   860.4        |   853.2        |   849.8        |   848.6        |
| 4 |   936.0        |   927.8        |   923.3        |   921.1        |

State beats copy (632.3/925.3/1123.1/1238.3), raw-DINO (562.1/779.6/912.6/986.8), and mean-pool (649.1/788.8/880.8/936.3) at every (K, k). Copy/raw/mean baselines are identical across runs — probe is on frozen SSv2 val features.

**Trained MLP heads (wandb `eval/pred_loss_k{k}`, final epoch).**

| k | `2ldiw9xk` | `7zotsrwf` | `jk05gf14` | `hp9v42d1` |
|---|------------|------------|------------|------------|
| 1 |   513.0    |   515.6    |   521.0    |   525.6    |
| 2 |   —        |   720.4    |   720.7    |   722.6    |
| 3 |   —        |   —        |   852.7    |   852.1    |
| 4 |   —        |   —        |   —        |   935.1    |

**Takeaways.**

- **Long-horizon state improves monotonically with K** on the ridge probe: k=4 drops 936.0 → 921.1 (−14.9), k=3 drops 860.4 → 848.6 (−11.8). Multi-horizon supervision shapes the state.
- **Small k=1 penalty** (+9.5 ridge, +12.6 trained-head from K=1 → K=4). Uniform 1/K weighting dilutes k=1 gradient by factor K; recoverable later with non-uniform weights.
- **Trained heads beat ridge probe at k=1,2; ridge ≥ trained at k=3,4.** Trained MLP is strictly more expressive than a linear map, so at k=3,4 the K=4 head is underfit under uniform weighting + 100 epochs — not a capacity ceiling.
- **A2 partially contradicts B1's "state is fine, rollout is the bottleneck" framing.** The state *can* be pushed further via multi-horizon training alone; A2 is not redundant. A1 (explicit forward model) remains the right next step for genuine iterated rollout, but A2 is a working direct-decode alternative for k≤2.

**Reproduce eval.**
```bash
# Ridge probe
PYTHONPATH=/home/manu/vjepa2 python root/evals/multi_horizon_probe.py \
    --checkpoint /nas/manu/vjepa2/outputs/pred_in_dino_space_mh_k<K>_<wandbID>/last.pt \
    --data_dir /nas/manu --max_horizon 4 --batch_size 128 --gpu 0

# Trained-head wandb numbers
.venv/bin/python scripts/pull_mh_wandb_cls.py
```

Patches runs (`97x4ktzc` K=2, `0hymll1d` K=3, `zn1nvup2` K=4) still in flight; will log separately when complete.

### 2026-04-18 — A2 multi-horizon training, Patches K-ablation (mid-training snapshot, epoch ~24/100)

Parallel patches runs for the A2 sprint (config mirrors `e6esmgmu` + `encoder.rnn.max_horizon=K`, batch_size 8, `--use_patch_tokens`). All 3 still training at time of eval; `last.pt` snapshotted to `/nas/manu/vjepa2/outputs/snapshots_2026-04-18/` to avoid read-during-write races, then evaluated. **Not final** — will re-run at epoch 100.

**Runs.**

| K | wandbID       | run_name                         | snapshot checkpoint                                                                                |
|---|---------------|----------------------------------|----------------------------------------------------------------------------------------------------|
| 2 | `97x4ktzc`    | `patch_pred_dino_space_mh_k2`    | `/nas/manu/vjepa2/outputs/snapshots_2026-04-18/patch_pred_dino_space_mh_k2_97x4ktzc_last.pt`       |
| 3 | `0hymll1d`    | `patch_pred_dino_space_mh_k3`    | `/nas/manu/vjepa2/outputs/snapshots_2026-04-18/patch_pred_dino_space_mh_k3_0hymll1d_last.pt`       |
| 4 | `zn1nvup2`    | `patch_pred_dino_space_mh_k4`    | `/nas/manu/vjepa2/outputs/snapshots_2026-04-18/patch_pred_dino_space_mh_k4_zn1nvup2_last.pt`       |

**Multi-horizon probe (ridge on frozen `state_t`).** K=1 reference `e6esmgmu` is fully trained (100 epochs); K>1 are at epoch ~24.

| k | `e6esmgmu` K=1 | `97x4ktzc` K=2 | `0hymll1d` K=3 | `zn1nvup2` K=4 |
|---|----------------|----------------|----------------|----------------|
| 1 |   867.0        |   867.4        |   872.9        |   877.4        |
| 2 |  1100.2        |  1094.6        |  1094.8        |  1095.1        |
| 3 |  1220.0        |  1212.1        |  1209.6        |  1207.8        |
| 4 |  1279.8        |  1271.8        |  1268.6        |  1265.9        |

Baselines (identical across K): copy 1124.6/1525.0/1755.2/1875.0; raw-DINO 919.7/1160.7/1279.8/1336.6; mean-pool 993.1/1144.6/1229.1/1274.5.

**Trained MLP heads (wandb `eval/pred_loss[_k{k}]`, most recent step).**

| k | `e6esmgmu` (final) | `97x4ktzc` (ep~24) | `0hymll1d` (ep~24) | `zn1nvup2` (ep~24) |
|---|--------------------|--------------------|--------------------|--------------------|
| 1 |   851.5            |   858.1            |   862.7            |   865.8            |
| 2 |   —                |  1095.2            |  1096.9            |  1094.0            |
| 3 |   —                |   —                |  1222.3            |  1219.0            |
| 4 |   —                |   —                |   —                |  1288.1            |

**Takeaways (tentative — mid-training).**

- Same qualitative pattern as CLS: monotone long-horizon state improvement with K (k=4: −13.9, k=3: −12.2), small k=1 penalty (+10.4). Signal visible at epoch 24 already; expect it to strengthen at convergence.
- Mean-pool is a stronger patches baseline than raw-DINO at k=2,3,4 (per-token averaging is informative when most patches are static background). State still beats mean-pool at every (K, k).
- Ridge probe ≤ trained MLP head at k=1,2 but ridge beats trained head at k=3,4 — same underfitting signature as CLS. Uniform 1/K weighting + limited epochs leaves long-horizon head capacity on the table.

**Reproduce.** See CLS section above for commands; swap `<wandbID>` and use the snapshot path for patches. When training finishes, re-probe on the live `last.pt` under each run's output dir.

### 2026-04-19 — Multi-horizon probe with `lin_reg_history` (full-T left-padded concat)

Extended `multi_horizon_probe.py` to add a 4th baseline: `lin_reg_history = concat_leftpad(x_0..x_t) @ W_hist`, input dim T·D = 3072 (with left-zero-padding so `x_t` always occupies the last D-slot). Strictly dominates `lin_reg_last` and `mean-pool` as a linear probe, so the full decomposition

```
copy − state = (copy − raw) + (raw − mean) + (mean − hist) + (hist − state)
                LIN PROJ       AGGREG         LIN DYNAMICS    NONLIN DYNAMICS
```

has a known sign on LIN PROJ (≥ 0), LIN DYN (≥ 0), and lets NONLIN DYN (`hist − state`) isolate exactly "does the RNN's nonlinearity beat a linear AR model." AGGREG can still flip (raw vs mean aren't nested). Baseline numbers `copy`, `raw`, `mean`, `state` reproduce the 2026-04-09 entry bit-identically — no regression from the additive code change.

**CLS — `2ldiw9xk`** (`pred_in_dino_space_2ldiw9xk/last.pt`):

| k | copy   | raw    | mean   | hist   | state  | LIN PROJ | AGGREG | LIN DYN | NONLIN DYN |
|---|--------|--------|--------|--------|--------|----------|--------|---------|------------|
| 1 |  632.3 |  562.1 |  649.1 |  528.1 |  524.0 |    70.1  |  −87.0 |  121.0  |      4.1   |
| 2 |  925.3 |  779.6 |  788.8 |  728.8 |  730.2 |   145.7  |   −9.2 |   60.0  |     −1.4   |
| 3 | 1123.1 |  912.6 |  880.8 |  853.7 |  860.4 |   210.5  |   31.9 |   27.1  |     −6.7   |
| 4 | 1238.3 |  986.8 |  936.3 |  929.9 |  936.0 |   251.5  |   50.5 |    6.4  |     −6.0   |

**Takeaways (CLS).**
- LIN PROJ grows monotonically with k (mean-regression dominates at long k).
- AGGREG flips sign around k=2→3: averaging hurts when x_t is already near-optimal (short k), helps once x_t decorrelates (long k).
- LIN DYN collapses with k (121 → 6): the ordering-specific linear signal is concentrated at k=1 and essentially gone by k=4.
- **NONLIN DYN ≈ 0 and goes negative for k≥2**: the RNN state is at best marginally better than a linear AR model over ordered past frames, and slightly *worse* at k=2,3,4. The RNN's nonlinearity buys nothing beyond `lin_reg_history`.

**Patches — `e6esmgmu`** (`patch_pred_dino_space_e6esmgmu/last.pt`):

| k | copy   | raw    | mean   | hist   | state  | LIN PROJ | AGGREG | LIN DYN | NONLIN DYN |
|---|--------|--------|--------|--------|--------|----------|--------|---------|------------|
| 1 | 1124.6 |  919.7 |  993.1 |  868.9 |  867.0 |   204.8  |  −73.4 |  124.2  |      2.0   |
| 2 | 1525.0 | 1160.7 | 1144.6 | 1093.8 | 1100.2 |   364.3  |   16.0 |   50.8  |     −6.4   |
| 3 | 1755.2 | 1279.8 | 1229.1 | 1208.5 | 1220.0 |   475.4  |   50.7 |   20.6  |    −11.6   |
| 4 | 1875.0 | 1336.6 | 1274.5 | 1269.8 | 1279.8 |   538.4  |   62.1 |    4.7  |    −10.0   |

**Takeaways (patches).**
- Same qualitative pattern as CLS, larger magnitudes throughout. Reproduction matches the 2026-04-09 entry bit-for-bit on the existing columns.
- LIN DYN collapses from 124 (k=1) to 4.7 (k=4) — by k=4, `hist` (ordered linear AR) and `mean-pool` (unordered) differ by only ~5 units. For patches, ordering buys almost nothing at long horizons — most patches are static background, where the RNN's "ordered" advantage degenerates into "averaging."
- **NONLIN DYN is more negative than CLS** at every k≥2 (−6.4 / −11.6 / −10.0 vs CLS −1.4 / −6.7 / −6.0). The patches RNN is *farther* below the linear-AR ceiling than the CLS RNN.

**Joint punchline (CLS + patches).** At every horizon k≥2, neither RNN beats the linear-AR ceiling `lin_reg_history`. Whatever multi-step information ends up in `state_t` is no better than what a closed-form linear regression over `concat_leftpad(x_0..x_t)` extracts directly from raw DINO features. The earlier "AR rollout collapses past k=1" finding was framed as a rollout-mechanism failure; the multi-horizon decomposition shows the state itself isn't carrying anything beyond a linear function of past frames, so the issue isn't *just* the iteration mechanism — it's that the trained nonlinear surprise update isn't producing more useful representations than a stack of past frames + a linear map. Argues that A1 (explicit forward model) is necessary but not sufficient: the *encoder* also needs an objective that pressures it past the linear ceiling.

**Run commands.**
```bash
# CLS (GPU 0, ~10 min)
PYTHONPATH=/home/manu/vjepa2 /home/manu/vjepa2/.venv/bin/python \
    root/evals/multi_horizon_probe.py \
    --checkpoint /nas/manu/vjepa2/outputs/pred_in_dino_space_2ldiw9xk/last.pt \
    --data_dir /nas/manu --max_horizon 4 --batch_size 128 --gpu 0

# Patches (GPU 1, ~4 h)
PYTHONPATH=/home/manu/vjepa2 /home/manu/vjepa2/.venv/bin/python \
    root/evals/multi_horizon_probe.py \
    --checkpoint /nas/manu/vjepa2/outputs/patch_pred_dino_space_e6esmgmu/last.pt \
    --data_dir /nas/manu --max_horizon 4 --batch_size 32 --gpu 1
```

Logs: `outputs/mh_probe_{2ldiw9xk,e6esmgmu}_fullT_20260419.log`.

### 2026-04-19 — MLP probe (evalID: `mlp_probe`) — CLS Tier 1

Tier 1 test of Poss 1 (linear AR is the ceiling) vs Poss 2 (nonlinear structure exists but RNN misses it) per `docs/meet2.md` TAKEAWAY SO FAR. MLP fit on frozen `state_t` and `concat_history` with SGD. Implementation: residual MLP (linear skip + zero-init GELU branch), skip warm-started with closed-form ridge and **frozen** during SGD — so the nonlinear branch only needs to find corrections on top of the ridge solution. Guarantees final val ≤ ridge (nonlinear can always stay at zero) so no under-fit risk.

**CLS — `2ldiw9xk`** (`pred_in_dino_space_2ldiw9xk/last.pt`)

Hyperparameters: `--mlp_hidden 1024 --mlp_layers 2 --epochs 20 --lr 3e-4 --weight_decay 1e-4 --patience 5 --seed 0 --batch_size 128`. Ridge lambda 1e-3 (default). Training time ~8 min on a single GPU (all probes early-stopped by epoch 8 of 20).

| k | ridge(state) | mlp(state) | Δstate | ridge(hist) | mlp(hist) | Δhist | Δhist % |
|---|--------------|------------|--------|-------------|-----------|-------|---------|
| 1 | 524.0        | 520.2      |  3.8   | 528.1       | 524.9     |  3.2  | 0.61%   |
| 2 | 730.2        | 723.5      |  6.7   | 728.8       | 724.1     |  4.7  | 0.65%   |
| 3 | 860.4        | 852.3      |  8.1   | 853.7       | 848.1     |  5.6  | 0.66%   |
| 4 | 936.0        | 928.6      |  7.4   | 929.9       | 924.3     |  5.6  | 0.60%   |

Sanity gate: `mlp(hist, k=1) = 524.9 ≤ ridge(hist, k=1) = 528.1` ✓ (MLP strictly more expressive; warm-start + frozen skip + zero-init nonlinear guarantees this by construction).

**Decision: Possibility 1 for CLS.** Δhist is 0.60-0.66% across all k — within the ~1% noise band of Poss 1 (decision rule in `delete_plan_MLP_probe.md`). Even with access to a nonlinear function class, the MLP can only shave ~0.6% off the ridge ceiling on the raw 8 DINO frames. The linear-AR figure in Tables 1/2 of `meet2.md` is (to within 1%) the real extractable ceiling — not a probe-class artifact.

**Δstate ≈ Δhist** (0.6-0.9% on state, 0.6% on hist). The nonlinear branch finds essentially the same small residual when applied to the RNN state as to the raw concat. Consistent with Poss 1 reading: there isn't significantly more nonlinear structure in the raw frames than what the state already encodes linearly.

**Notes on the SGD path** (lessons learned; retained here because they affect reproducibility):
- First attempt (plain ReLU MLP, no warm-start, LR 1e-3, patience 3) plateaued ~30% above ridge on `concat_history` within 7 epochs. Confused early stopping with "convergence". Discarded.
- Residual design (skip + zero-init nonlinear) fixed init: forward(x) = ridge(x) at start.
- With skip trainable + LR 1e-3, SGD drifted skip away from ridge on concat_history (1.2M-param 3072→384 linear overshoots with batched-per-t updates). Freezing skip after warm-start + batched-per-t loss + LR 3e-4 made training stable.

**Run command**:
```bash
bash scripts/repro/mlp_probe_tier1.sh    # launches CLS (GPU 0) + patches (GPU 1)
```

Logs: `outputs/mlp_probe_tier1/{2ldiw9xk,e6esmgmu}.log`.

### 2026-04-20 — MLP probe (evalID: `mlp_probe`) — Patches Tier 1

Same hyperparameters and implementation as the CLS Tier 1 run above (residual MLP with ridge warm-start + frozen skip; `--mlp_hidden 1024 --mlp_layers 2 --epochs 20 --lr 3e-4 --weight_decay 1e-4 --patience 5 --seed 0 --batch_size 32`). Per-token probe: one shared MLP across 256 spatial positions, matching the ridge convention.

**Patches — `e6esmgmu`** (`patch_pred_dino_space_e6esmgmu/last.pt`)

Training wall-clock: warm-start ridge accumulator pass ≈ 3.5 h (dominated by fp64 3072×3072 matmul on 5279 batches), MLP training ≈ 2 h. All probes early-stopped by epoch ~15 of 20.

| k | ridge(state) | mlp(state) | Δstate | ridge(hist) | mlp(hist) | Δhist | Δhist % |
|---|--------------|------------|--------|-------------|-----------|-------|---------|
| 1 | 867.0        | 848.7      |  18.3  | 868.9       | 847.3     | 21.6  | **2.49%** |
| 2 | 1100.2       | 1075.6     |  24.6  | 1093.8      | 1072.3    | 21.5  | **1.96%** |
| 3 | 1220.0       | 1192.3     |  27.7  | 1208.5      | 1187.3    | 21.2  | **1.75%** |
| 4 | 1279.8       | 1252.3     |  27.5  | 1269.8      | 1248.2    | 21.6  | **1.70%** |

Sanity gate: `mlp(hist, k=1) = 847.3 ≤ ridge(hist, k=1) = 868.9` ✓.

**Decision: Possibility 2 for patches (weakly).** Δhist is 1.70-2.49%, crossing the ≥2% Poss 2 threshold at k=1 and sitting in the ambiguous 1-2% band for k=2,3,4. Our 2-layer MLP finds nonlinear structure in raw patch DINO concat that ridge misses — in contrast to CLS, where Δhist was 0.60-0.66% (clean Poss 1).

**Patches vs CLS — why they differ.** Patches preserve per-token spatial features. Local motion / object dynamics can show up as conditional, position-dependent patterns that a linear map cannot express but a 2-layer ReLU/GELU MLP can. CLS is a single aggregated semantic vector per frame — any such nonlinear spatial patterns are averaged out before the probe ever sees them, leaving only globally-linear structure.

**Δstate ≈ Δhist** across k. The RNN state (K=1 trained) has captured essentially the same nonlinear content a fresh MLP-on-concat finds. Consistent with the earlier reading that K=1 training already surfaces most linearly-accessible multi-step info.

**Combined CLS + Patches picture.**
- CLS: linear AR ≈ nonlinear MLP ≈ our RNN state. Pinpointed ceiling. Poss 1.
- Patches: nonlinear MLP beats ridge by ~2%. Real structure; our K=1 state is within noise of it; the data has more.
- Token type matters a lot for whether nonlinear probing helps. The 2-layer 1024-hidden MLP isn't at the true nonlinear ceiling on patches; a bigger probe (Tier 3 capacity ablation) could widen the gap further and quantify the headroom.

**Sharp follow-up now motivated.** On the A2 K=4 patches run (`zn1nvup2`, once it finishes training): compute `mlp_probe` with the same hyperparameters. If A2 K=4 state beats `mlp(hist)` on patches by more than 2% at k≥2, multi-horizon supervision is genuinely pushing the state past the nonlinear-MLP-on-concat ceiling. If A2 state ≈ `mlp(hist)`, A2's K=4 gain is just matching what a fresh MLP probe already extracts. Current A2 K=4 patches snapshot (epoch ~24) ridge-probe gave state = 1265.9 at k=4; `mlp(hist, k=4) = 1248.2`, so the snapshot is ~1.4% ABOVE `mlp(hist)`. Final converged number is the one to read.

**Tier 2 / Tier 3 decision.** Patches Δhist at 1.70-2.49% is marginal. A capacity ablation (Tier 3 — 3 MLP sizes at k=4 on patches) would tighten whether the true ceiling is meaningfully below ridge or just barely. Tier 2 (MLP on `last` / `meanpool`) would tell us where the nonlinear gain lives (single-frame appearance vs aggregation vs ordered history). Both worthwhile on patches; skip for CLS (clear Poss 1).

**Run command**:
```bash
bash scripts/repro/mlp_probe_tier1.sh    # launches CLS (GPU 0) + patches (GPU 1)
```

Logs: `outputs/mlp_probe_tier1/{2ldiw9xk,e6esmgmu}.log`.

### 2026-04-20 — Static/Dynamic decomposition on Causal Transformer patches (`r55x2lcn`)

Extended Table 6 in `docs/meet1.md` to include a causal transformer row so the patch RNN-vs-transformer gap (62 L2 total from Table 3) can be split into dynamic vs static contributions. Ran `root/evals/static_dynamic_decomposition.py` unchanged — the script reads `model.pred_error_l2`, which `CausalTransformerPredictor` already exposes in the same `(B, T, S)` format as the RNN (`root/models/causal_transformer.py:82-88`; `root/models/model.py:127-131` sets the attribute identically for both encoder types).

**Table 6 (extended).** Copy baselines are data-only (identical to the `e6esmgmu` row).

| Group            | Copy   | RNN (`e6esmgmu`)  | Causal Transformer (`r55x2lcn`) |
|------------------|--------|-------------------|---------------------------------|
| Dynamic patches  | 1430.6 | 1077.0 (**24.7%**) | 984.6 (**31.2%**)              |
| Static patches   | 746.9  | 625.9 (**16.2%**) | 592.6 (**20.7%**)              |

Dynamic-only copy shuffle ratio (data property, unchanged): 1.40x.

**Interpretation.**
- Both archs show the same dynamic-bias ratio (RNN 1.52x, Transformer 1.51x) — neither is uniquely motion-aware; both lift dynamic patches proportionally more than static.
- Transformer's absolute gap over RNN: dynamic Δ = 92.4 L2, static Δ = 33.3 L2. ≈73% of the total patch gap lives on dynamic patches. Improving RNN's dynamic-patch modeling is the higher-leverage lever for closing Table 3.
- Caveat: the 62-L2 patch gap in Table 3 is confounded by capacity (transformer ≈ 7.1M vs RNN ≈ 440K params) and by cross-token attention. Static/dynamic split diagnoses *where* the gap is, not *why*. See `docs/meet2.md` "RNN vs Causal Transformer — decoder and encoder expressivity" for the architectural audit.

**Reproduce.**
```bash
PYTHONPATH=/home/manu/vjepa2 python root/evals/static_dynamic_decomposition.py \
    --checkpoint /nas/manu/vjepa2/outputs/causal_pred_patches_r55x2lcn/last.pt \
    --data_dir /nas/manu --batch_size 32 --gpu 0
```

Log: `outputs/static_dynamic_r55x2lcn.log`.

## Open Questions
-
