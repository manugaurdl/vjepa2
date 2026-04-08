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
