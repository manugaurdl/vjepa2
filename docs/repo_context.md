I am building upon vjepa's repository. The entry point of the codebase is train.py. My codebase utilizes functionalities from vjepa's repo (src/). The code I have written resides insied root/.
I am training a RNN on something something v2 dataset.

## Running scripts
- Always set `PYTHONPATH=/home/manu/vjepa2` when running scripts that import from `root/` (e.g. `PYTHONPATH=/home/manu/vjepa2 python root/evals/temporal_shuffle_test.py ...`)
- For wandb API calls, use `wandb.Api(timeout=60)` — the default 19s timeout is too short and causes `ReadTimeoutError`

## Eval: UCF101 Linear Probe (evalID: `ucf101_probe`)

Reproduces **Table 1** in `docs/april_3_meet.md`. 20 epochs, AdamW lr=1e-3, batch=64, no aug, features cached after epoch 0 then DINO freed from GPU. ~3-5 min per row on a free RTX 6000 Ada.

UCF101 data: `/nas/manu/ucf101/data/{train,test}.csv` (absolute paths inside CSV — no `--data_dir` needed). Always `source /home/manu/vjepa2/.venv/bin/activate` first.

### Reproducing the full table

```bash
bash scripts/repro/table1_ucf101_probe.sh   # all 6 rows in parallel on GPUs 0-5, ~5 min
```

The script prints a summary and writes results to `outputs/repro_table1/<name>/probe_results.json` (use `best_acc`). Per-row checkpoint mapping is hard-coded inside the script — see the header comment.

### Per-model checkpoint mapping

The `best.pt` vs `last.pt` distinction matters (this caused Table 2 to be wrong — see CLAUDE.md). Use exactly:

| wandbID | Checkpoint |
|---|---|
| `zyvsy8gk` (CLS, CE+pred, learned)   | `outputs/update=w(error)_L2weight1e-1_zyvsy8gk/best.pt` |
| `2ldiw9xk` (CLS, pred, DINO)         | `outputs/pred_in_dino_space_2ldiw9xk/last.pt` *(no best.pt)* |
| `e6esmgmu` (Patch, pred, DINO)       | `outputs/patch_pred_dino_space_e6esmgmu/last.pt` *(no best.pt)* |
| `tj9x820q` (Patch, CE+pred, learned) | `outputs/patch_ce_pred_tj9x820q/best.pt` |

Paths relative to `/nas/manu/vjepa2/`. The script auto-detects `use_patch_tokens` from `ckpt["args"]`, so the same `eval_transfer.py` invocation works for CLS and patch checkpoints.

### Single-row template (when you only need one)

```bash
PYTHONPATH=/home/manu/vjepa2 python eval_transfer.py \
    --checkpoint <abs_path_to_ckpt.pt> \
    --mode probe --train_csv /nas/manu/ucf101/data/train.csv \
    --data_csv /nas/manu/ucf101/data/test.csv \
    --num_classes 101 --cache_features --no_aug \
    --output_dir outputs/eval_ucf101_<wandbID>
```

Add `--baseline` for DINO mean-pool or `--concat` for DINO concat. **Gotcha:** in baseline/concat mode `--checkpoint` is required by argparse but its contents are unused — pass any valid `.pt` as a placeholder. **Gotcha:** the path `update=w(error)_L2weight1e-1_zyvsy8gk` contains parentheses — escape them in bash (`\(`, `\)`) or single-quote the whole path.

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
- Cached features: `/nas/manu/ssv2/dino_feats/vits14/{validation,validation_patches}.pt`.
  - `validation_patches.pt` correctly sized `(24777, 8, 256, 384)` (built by `precompute_patch_feats.py`).
  - **`validation.pt` (CLS) is zero-padded to `(168913, 8, 384)`** — shape matches the train split, not val. This is a historical artifact: `root/models/model.py:74` allocates `id_to_feat = zeros(168913, 8, 384)` regardless of split, and the legacy `train.py --cache_dino_feats` save site wrote the whole buffer. The first 24777 rows are real, the rest are zeros. **Any reader that loads `validation.pt` directly MUST truncate to the row count of `ssv2/data/validation.csv` (currently 24777) before averaging** — otherwise `feats.mean()` is diluted ~7×, which silently breaks pred_loss numbers (this caused the wrong Table 2 figures of Transformer=100.1 / RNN=176.9; correct are 517 / 513). `temporal_shuffle_test.py` and `static_dynamic_decomposition.py` truncate internally; `train.py`'s val loop is fine (the dataset iterates only real CSV rows). The save sites in `train.py` now trim with `model.id_to_feat[:len(loader.dataset)]`, so any newly cached `validation.pt` will be correctly sized.
- Checkpoints at `/nas/manu/vjepa2/outputs/<run_name>/{best,last}.pt`. Contain `model` state dict and `args` config dict.