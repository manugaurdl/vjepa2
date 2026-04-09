"""
Autoregressive rollout eval: does the model extrapolate trajectories, or does it
collapse to a constant once fresh frames stop arriving?

Procedure (per video):
  1. Build the RNN state on the first `t_ctx` frames (observed context).
  2. From the resulting state, iterate `w_pred` K = T - t_ctx times without
     feeding new frames: pred_1 = w_pred(state), pred_2 = w_pred(w_pred(state)), ...
  3. Compare each pred_k to the ground-truth DINO feature at the corresponding
     horizon, using sum-over-D L2 (same unit as Tables 2/3 / copy baseline).

Baselines (per horizon k, averaged over val set):
  - Copy / drift curve:   ||x[t_ctx-1] - x[t_ctx-1+k]||^2
      This is a pure data property. An EMA or any constant predictor has no way
      to extrapolate once frames stop arriving, so it sits on this curve. The
      model needs to beat it to claim it learned dynamics.
  - Linear extrapolation: ||x[t_ctx-1] + k*(x[t_ctx-1] - x[t_ctx-2]) - x[t_ctx-1+k]||^2
      Constant-velocity prediction in DINO space. A "dumb dynamics" baseline
      sitting above the copy floor.

Interpretation:
  - Oracle lives at 0.
  - Copy curve is the floor for anything non-extrapolating (constant, EMA, smoother).
  - A dynamics model should sit between the two.
  - model >= copy at horizon k => the model did not extrapolate at that horizon.

Restrictions:
  - RNN encoder only (model.encoder must be VideoRNNTransformerEncoder, with
    core.update_type == "surprise" so core.w_pred exists).
  - DINO-space only (predict_in_dino_space=True). In a learned h-space, w_pred
    outputs live in h-space, not comparable to raw DINO features -> we skip.
  - Works for CLS (S=1), patch (S=256), and meanpool-patch (S=1) checkpoints;
    the feature-file suffix is picked the same way as temporal_shuffle_test.py.

Usage:
    PYTHONPATH=/home/manu/vjepa2 python root/evals/autoregressive_rollout.py \
        --checkpoint /nas/manu/vjepa2/outputs/<run>/last.pt \
        --data_dir /nas/manu \
        --t_ctx 4 --batch_size 64 --gpu 0
"""

import argparse
import os
import torch
from tqdm import tqdm

from root.models.model import _build_model
from root.utils import dict_to_namespace


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = dict_to_namespace(ckpt["args"])
    args.cache_dino_feats = False
    args.load_cache_feats = True
    args.val_dataset_len = None
    model = _build_model(args, device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, args


def check_supported(model, model_args):
    enc_type = model_args.encoder.type
    if enc_type != "rnn":
        raise RuntimeError(
            f"autoregressive_rollout currently supports encoder.type='rnn' only (got '{enc_type}'). "
            "Causal transformer rollout requires append-token mechanics — out of scope for v1."
        )
    predict_in_dino_space = getattr(model_args.encoder.rnn, "predict_in_dino_space", False)
    if not predict_in_dino_space:
        raise RuntimeError(
            "autoregressive_rollout requires predict_in_dino_space=True. In learned h-space, "
            "w_pred outputs live in the encoder's projected space and cannot be compared to "
            "raw DINO features — drop copy baseline would be meaningless."
        )
    core = model.encoder.core
    if core.update_type != "surprise":
        raise RuntimeError(f"Expected core.update_type='surprise', got '{core.update_type}'.")
    if not hasattr(core, "w_pred"):
        raise RuntimeError("core has no w_pred module — cannot iterate predictions.")


def pick_feat_suffix(model_args):
    use_patches = getattr(model_args, "use_patch_tokens", False)
    meanpool_patches = getattr(model_args, "meanpool_patches", False)
    if meanpool_patches:
        return "_meanpool"
    if use_patches:
        return "_patches"
    return ""


@torch.no_grad()
def rollout_batch(model, x_ctx, K):
    """
    Run RNN on context frames, then iterate w_pred K times.

    Args:
        model: DinoFrameEncoder with RNN encoder.
        x_ctx: (B, t_ctx, ...) context features (D or S,D on last dims).
        K:     number of rollout steps.

    Returns:
        preds: list of K tensors, each same shape as one frame slice of x_ctx
               (i.e. (B, D) for CLS/meanpool, (B, S, D) for patches).
    """
    # encoder.forward handles both (B,T,D) and (B,T,S,D) shapes.
    outs, final_state, *_ = model.encoder(x_ctx)
    # final_state: (B, D) if input was (B,T,D); (B, S, D) if input was (B,T,S,D).
    state_for_pred = final_state
    squeezed_cls = (state_for_pred.dim() == 2)
    if squeezed_cls:
        state_for_pred = state_for_pred.unsqueeze(1)  # (B, 1, D) — w_pred is token-wise

    w_pred = model.encoder.core.w_pred
    preds = []
    cur = state_for_pred
    for _ in range(K):
        cur = w_pred(cur)  # (B, S, D)
        out = cur.squeeze(1) if squeezed_cls else cur
        preds.append(out)
    return preds


def l2_sum_over_d(a, b):
    """||a - b||^2 summed over the last (feature) dim.

    a, b: (..., D). Returns (...) tensor.
    """
    return ((a - b) ** 2).sum(dim=-1)


@torch.no_grad()
def run_eval(model, feats, t_ctx, batch_size, device):
    """
    Loop over val set, accumulating per-horizon sums for:
      - model rollout error
      - copy baseline error
      - linear extrapolation baseline error

    Returns dict: horizon k -> {"model": float, "copy": float, "linear": float}
    """
    N, T = feats.shape[0], feats.shape[1]
    K = T - t_ctx  # number of rollout steps we can evaluate
    if K <= 0:
        raise ValueError(f"t_ctx={t_ctx} >= T={T}; no rollout horizons to evaluate.")
    if t_ctx < 2:
        raise ValueError("t_ctx must be >= 2 (linear baseline needs two context frames).")

    # Per-horizon running sums and sample counts.
    # For patch features, per-sample error is mean-over-S of sum-over-D L2 —
    # matching the Table 3 / static_dynamic convention.
    sum_model = [0.0] * K
    sum_copy = [0.0] * K
    sum_linear = [0.0] * K
    n_seen = 0

    for start in tqdm(range(0, N, batch_size), desc="AR rollout"):
        end = min(start + batch_size, N)
        x = feats[start:end].to(device, dtype=torch.float32)  # (B, T, ...) or (B, T, S, D)
        B = x.shape[0]

        x_ctx = x[:, :t_ctx]
        preds = rollout_batch(model, x_ctx, K)  # list of K, each (B, D) or (B, S, D)

        last_obs = x[:, t_ctx - 1]           # (B, ...) or (B, S, D)
        prev_obs = x[:, t_ctx - 2]           # for linear extrapolation
        velocity = last_obs - prev_obs       # same shape

        for k in range(1, K + 1):
            target = x[:, t_ctx - 1 + k]     # (B, ...) or (B, S, D)

            model_err = l2_sum_over_d(preds[k - 1], target)  # (B,) or (B, S)
            copy_err = l2_sum_over_d(last_obs, target)
            lin_pred = last_obs + k * velocity
            linear_err = l2_sum_over_d(lin_pred, target)

            if model_err.dim() == 2:  # patch case: mean over S tokens per sample
                model_err = model_err.mean(dim=-1)
                copy_err = copy_err.mean(dim=-1)
                linear_err = linear_err.mean(dim=-1)

            sum_model[k - 1] += model_err.sum().item()
            sum_copy[k - 1] += copy_err.sum().item()
            sum_linear[k - 1] += linear_err.sum().item()

        n_seen += B

    results = {}
    for k in range(1, K + 1):
        results[k] = {
            "model": sum_model[k - 1] / n_seen,
            "copy": sum_copy[k - 1] / n_seen,
            "linear": sum_linear[k - 1] / n_seen,
        }
    return results


def print_results(results, t_ctx):
    print()
    print("=" * 72)
    print(f"AUTOREGRESSIVE ROLLOUT  (context = first {t_ctx} frames)")
    print("=" * 72)
    print(f"{'horizon k':>10s} {'copy':>12s} {'linear':>12s} {'model':>12s} {'model/copy':>14s}")
    print("-" * 72)
    for k, row in results.items():
        ratio = row["model"] / max(row["copy"], 1e-8)
        marker = "  ← extrap" if ratio < 1.0 else ("  ← collapse" if ratio >= 1.0 else "")
        print(f"{k:>10d} {row['copy']:>12.2f} {row['linear']:>12.2f} {row['model']:>12.2f} {ratio:>13.3f}x{marker}")
    print()
    # Summary verdict.
    any_beats_copy = any(row["model"] < row["copy"] for row in results.values())
    all_beats_copy = all(row["model"] < row["copy"] for row in results.values())
    if all_beats_copy:
        print("→ Model beats copy at every horizon: extrapolates dynamics.")
    elif any_beats_copy:
        print("→ Model beats copy at some horizons but not all — partial extrapolation.")
    else:
        print("→ Model never beats copy: collapses to constant / behaves like a smoother.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--t_ctx", type=int, default=4, help="Number of context frames before rollout starts.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    model, model_args = load_model(args.checkpoint, device)
    check_supported(model, model_args)

    # Load cached val features (same logic as temporal_shuffle_test.py).
    suffix = pick_feat_suffix(model_args)
    dino_name = model_args.dino_model.split("_")[-1]
    feat_path = os.path.join(args.data_dir, "ssv2/dino_feats", dino_name, f"validation{suffix}.pt")
    print(f"Loading features from {feat_path}")
    feats = torch.load(feat_path, mmap=True)
    print(f"Feature shape: {tuple(feats.shape)}, dtype: {feats.dtype}")

    # Truncate padded CLS/meanpool cache to real val size (see CLAUDE.md / repo_context.md).
    val_csv = os.path.join(args.data_dir, "ssv2/data/validation.csv")
    n_valid = sum(1 for _ in open(val_csv))
    if feats.shape[0] > n_valid:
        print(f"Truncating padded cache: {feats.shape[0]} -> {n_valid} (real val samples)")
        feats = feats[:n_valid]

    results = run_eval(model, feats, args.t_ctx, args.batch_size, device)
    print_results(results, args.t_ctx)


if __name__ == "__main__":
    main()
