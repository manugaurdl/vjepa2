I am building upon vjepa's repository. The entry point of the codebase is train.py. My codebase utilizes functionalities from vjepa's repo (src/). The code I have written resides insied root/.
I am training a RNN on something something v2 dataset.

## Running scripts
- Always set `PYTHONPATH=/home/manu/vjepa2` when running scripts that import from `root/` (e.g. `PYTHONPATH=/home/manu/vjepa2 python root/evals/temporal_shuffle_test.py ...`)
- For wandb API calls, use `wandb.Api(timeout=60)` — the default 19s timeout is too short and causes `ReadTimeoutError`

## Eval: UCF101 Linear Probe

UCF101 data lives at `/nas/manu/ucf101/data/{train,test}.csv`. Checkpoints at `/nas/manu/vjepa2/outputs/<run_name>/last.pt`.

```bash
# RNN transfer probe
PYTHONPATH=/home/manu/vjepa2 python eval_transfer.py --checkpoint /nas/manu/vjepa2/outputs/<run_name>/last.pt \
    --mode probe --train_csv /nas/manu/ucf101/data/train.csv \
    --data_csv /nas/manu/ucf101/data/test.csv \
    --num_classes 101 --cache_features --no_aug --output_dir outputs/eval_ucf101_<run_name>

# DINO mean-pool baseline
PYTHONPATH=/home/manu/vjepa2 python eval_transfer.py --checkpoint /nas/manu/vjepa2/outputs/<run_name>/last.pt \
    --mode probe --baseline --train_csv /nas/manu/ucf101/data/train.csv \
    --data_csv /nas/manu/ucf101/data/test.csv \
    --num_classes 101 --cache_features --no_aug --output_dir outputs/eval_ucf101_baseline

# DINO concat baseline
PYTHONPATH=/home/manu/vjepa2 python eval_transfer.py --checkpoint /nas/manu/vjepa2/outputs/<run_name>/last.pt \
    --mode probe --concat --train_csv /nas/manu/ucf101/data/train.csv \
    --data_csv /nas/manu/ucf101/data/test.csv \
    --num_classes 101 --cache_features --no_aug --output_dir outputs/eval_ucf101_baseline_concat
```

## Eval: Temporal Shuffle Test

Tests whether model relies on temporal order. Shuffles frame order at eval time, compares pred_loss.

```bash
PYTHONPATH=/home/manu/vjepa2 python root/evals/temporal_shuffle_test.py \
    --checkpoint /nas/manu/vjepa2/outputs/<run_name>/best.pt \
    --data_dir /nas/manu --batch_size 64 --num_shuffles 3 --gpu 0
```

- Loads cached val features from `/nas/manu/ssv2/dino_feats/vits14/validation{_patches}.pt` based on `use_patch_tokens` in checkpoint config
- Outputs: normal pred_loss, shuffled pred_loss (avg over seeds), ratio
- Accesses `model.pred_error_l2` after forward pass (shape `(B, T, S)`, skips t=0)

## Eval: Static vs Dynamic Patch Decomposition

Splits patches into static/dynamic by per-patch motion score, compares model error vs copy baseline per group. Patch models only.

```bash
PYTHONPATH=/home/manu/vjepa2 python root/evals/static_dynamic_decomposition.py \
    --checkpoint /nas/manu/vjepa2/outputs/<run_name>/best.pt \
    --data_dir /nas/manu --batch_size 64 --gpu 0
```

- Copy baseline comparison only valid for DINO-space models (`predict_in_dino_space=True`). For learned-space models, ignore copy baseline column — model error per group is still meaningful.
- Also computes copy shuffle ratio on dynamic patches only (data-only metric, valid for any model).

## Key model internals

- `pred_error_l2`: stored on `model.pred_error_l2` after forward pass, shape `(B, T, S)` where S=256 for patches, S=1 for CLS. Computed as `(h - w_pred(state))^2.sum(dim=-1)` in `GatedTransformerCore.forward()` (`root/models/rnn.py:198-199`).
- State update (surprise mode): `error = h - w_pred(state)` → `update = w_precision(error)` → `state = LN(state + update)`
- `w_pred`: 2-layer MLP (Linear→ReLU→Linear), hidden_dim=384, defined in `root/models/rnn.py`
- Cached features: `/nas/manu/ssv2/dino_feats/vits14/{validation,validation_patches}.pt`. Patches shape: `(24777, 8, 256, 384)`, CLS shape: `(24777, 8, 384)`.
- Checkpoints at `/nas/manu/vjepa2/outputs/<run_name>/{best,last}.pt`. Contain `model` state dict and `args` config dict.