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

## Open Questions
-
