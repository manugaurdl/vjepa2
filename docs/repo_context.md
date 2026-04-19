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

- Loads cached val features from `/nas/manu/ssv2/dino_feats/vits14/validation{,_patches,_meanpool}.pt` based on `use_patch_tokens` / `meanpool_patches` flags in checkpoint config
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

## Eval: Autoregressive Rollout (evalID: `autoregressive_rollout`)

Tests whether the model extrapolates dynamics or collapses to a constant once fresh frames stop arriving. Runs the RNN on the first `t_ctx` frames to build a state, then iterates `w_pred` for K = T − t_ctx steps with no new input (literal `w_pred(state) → w_pred(w_pred(state)) → ...`). Compares each horizon against a copy baseline (the frame-drift curve — the floor for any non-extrapolating predictor, incl. EMAs) and a linear-extrapolation baseline.

```bash
PYTHONPATH=/home/manu/vjepa2 python root/evals/autoregressive_rollout.py \
    --checkpoint /nas/manu/vjepa2/outputs/<run_name>/last.pt \
    --data_dir /nas/manu --t_ctx 4 --batch_size 64 --gpu 0
```

- **RNN-only, DINO-space only.** Learned-space models (`predict_in_dino_space=False`) skipped: `w_pred` outputs live in the projected space and aren't comparable to raw DINO targets. Causal transformer rollout (append-token mechanics) is out of scope for v1.
- Supports CLS (S=1), patches (S=256), and meanpool-patch (S=1). Picks `validation{,_patches,_meanpool}.pt` based on `use_patch_tokens` / `meanpool_patches` in the checkpoint config. Truncates padded CLS/meanpool cache.
- Per-horizon error is sum-over-D L2 (same unit as Tables 2/3). For patch models, further averaged over S tokens per sample (matches `static_dynamic_decomposition.py`).
- Context length `t_ctx` must be ≥ 2 (linear baseline needs the last two observed frames to estimate velocity). Default 4 gives K = 4 rollout steps for the standard 8-frame SSv2 clips.
- Interpretation: model < copy at horizon k ⇒ extrapolating. model ≥ copy ⇒ collapsed to constant / behaving like a smoother. The copy curve is exactly what an EMA rolled out autoregressively would produce (it has nothing to integrate once frames stop coming, so it sits at the data's drift floor).

## Eval: Multi-Horizon Linear Probe (evalID: `multi_horizon_probe`)

Tests whether the RNN state `state_t` carries multi-step future information *beyond* what the current frame `x_t` trivially encodes. Fits closed-form ridge regression `state_t @ W_k ≈ x_{t+k}` for k=1..K on SSv2 train, evals per-horizon L2 on SSv2 val. Fits a parallel raw-DINO probe `x_t @ W_k_raw ≈ x_{t+k}` as the reference; the gap `raw - state` is exactly the encoder's multi-step contribution on top of linearly projecting the current frame forward.

```bash
PYTHONPATH=/home/manu/vjepa2 python root/evals/multi_horizon_probe.py \
    --checkpoint /nas/manu/vjepa2/outputs/<run_name>/last.pt \
    --data_dir /nas/manu --max_horizon 4 --batch_size 128 --gpu 0
```

- **RNN-only.** Causal transformer support not wired up in v1 (encoder.outs has the same shape semantics — would slot in identically).
- Works for CLS (S=1), patches (S=256), and meanpool-patch (S=1). For patches, one shared `W: D×D` is fit across all 256 spatial positions — token-wise, matching `w_pred`'s inductive bias. Cannot capture cross-patch motion (would need `(S·D)×(S·D)` ≈ 9B params).
- Learned-space and DINO-space checkpoints both supported: the linear probe subsumes any final projection.
- No SGD, no epochs, no tuned hyperparameters — one pass over train to accumulate `XtX`/`XtY` in fp64, one solve, one pass over val. Only knob is `--ridge_lambda` (relative: `reg = λ · trace(XtX) / D`, default 1e-3).
- Per-horizon error is sum-over-D L2 (same unit as Tables 2/3). Patch models: further averaged over S tokens per sample (matches `static_dynamic_decomposition.py` convention).
- `t` ranges over `1..T-k-1`: t=0 is skipped because state starts from zeros, so `state_0` is trivially empty.
- Use `--train_limit` for smoke tests (e.g. `--train_limit 2000` runs in ~1 min).
- Interpretation:
  - `state_probe < raw_probe` at all k → encoder put real multi-step info in the state beyond the current frame.
  - `state_probe ≈ raw_probe` → state ≈ current frame, no extra temporal signal.
  - `state_probe ≥ copy` → state encodes nothing useful (broken/noise).

## Training: Multi-horizon prediction heads (A2)

Trains the RNN to predict `x_{t+k}` for k=1..K from `state_t` via K separate MLP heads — the k=1 head (`w_pred`) still drives the surprise update; heads for k=2..K (`w_pred_extra: ModuleList`) only contribute to the loss. Loss is the uniform-weighted sum `(1/K)·Σ_k ||w_pred_k(state_t) − x_{t+k}||²` averaged over all valid (t, k) pairs. Config: `encoder.rnn.max_horizon: K` (default 1 → identical to prior behavior, bit-for-bit); `encoder.rnn.horizon_weights: null` (uniform) or list of length K.

```bash
# CLS
python train.py --config base --wandb.run_name pred_in_dino_space_mh_k<K> \
    --load_cache_feats --no-action_classification \
    --encoder.rnn.predict_in_dino_space --encoder.rnn.max_horizon <K> --epochs 100

# Patches (batch_size 8 due to S=256)
python train.py --config base --wandb.run_name patch_pred_dino_space_mh_k<K> \
    --load_cache_feats --use_patch_tokens --no-action_classification \
    --encoder.rnn.predict_in_dino_space --encoder.rnn.max_horizon <K> \
    --batch_size 8 --val_batch_size 8 --epochs 100
```

- `max_horizon=1` path is bit-identical to pre-A2 code (verified via strict=True checkpoint load of `2ldiw9xk`/`e6esmgmu` + reproducing Table 7). Old K=1 checkpoints load into the new model with no migration.
- Training returns 7-tuple from `VideoRNNTransformerEncoder.forward`: `(hidden_states, final_state, gate, update_norm, r_novelty, pred_error_l2, multi_horizon_errors)`. `multi_horizon_errors` is a list of length K with per-horizon (B, T−k, S) L2 tensors; `pred_error_l2` is unchanged (k=1 signal). Causal transformer returns `None` for the 7th element.
- Per-horizon val metrics logged as `eval/pred_loss_k{1..K}`; summed weighted scalar still as `eval/pred_loss` so existing dashboards don't break. Train-side `trainer/pred_loss_k{1..K}` added symmetrically.
- Evaluate trained checkpoints via `multi_horizon_probe` above (ridge probe on frozen state — measures state informativeness) and by pulling `eval/pred_loss_k{k}` from wandb (measures the trained MLP head). See `scripts/pull_mh_wandb_{cls,patches}.py`.

## Key model internals

- `pred_error_l2`: stored on `model.pred_error_l2` after forward pass, shape `(B, T, S)` where S=256 for patches, S=1 for CLS. Computed as `(h - w_pred(state))^2.sum(dim=-1)` in `GatedTransformerCore.forward()` (`root/models/rnn.py:198-199`).
- State update (surprise mode): `error = h - w_pred(state)` → `update = w_precision(error)` → `state = LN(state + update)`
- `w_pred`: 2-layer MLP (Linear→ReLU→Linear), hidden_dim=384, defined in `root/models/rnn.py`
- Cached features: `/nas/manu/ssv2/dino_feats/vits14/{validation,validation_patches}.pt`.
  - `validation_patches.pt` correctly sized `(24777, 8, 256, 384)` (built by `precompute_patch_feats.py`).
  - **`validation.pt` (CLS) is zero-padded to `(168913, 8, 384)`** — shape matches the train split, not val. This is a historical artifact: `root/models/model.py:74` allocates `id_to_feat = zeros(168913, 8, 384)` regardless of split, and the legacy `train.py --cache_dino_feats` save site wrote the whole buffer. The first 24777 rows are real, the rest are zeros. **Any reader that loads `validation.pt` directly MUST truncate to the row count of `ssv2/data/validation.csv` (currently 24777) before averaging** — otherwise `feats.mean()` is diluted ~7×, which silently breaks pred_loss numbers (this caused the wrong Table 2 figures of Transformer=100.1 / RNN=176.9; correct are 517 / 513). `temporal_shuffle_test.py` and `static_dynamic_decomposition.py` truncate internally; `train.py`'s val loop is fine (the dataset iterates only real CSV rows). The save sites in `train.py` now trim with `model.id_to_feat[:len(loader.dataset)]`, so any newly cached `validation.pt` will be correctly sized.
- Checkpoints at `/nas/manu/vjepa2/outputs/<run_name>/{best,last}.pt`. Contain `model` state dict and `args` config dict.