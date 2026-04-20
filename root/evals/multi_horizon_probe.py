"""
Multi-horizon linear probe from RNN state (evalID: `multi_horizon_probe`).

Tests whether the frozen encoder's state `state_t` carries multi-step future
information beyond what the current frame `x_t` trivially encodes. Does not
touch training or require the architecture to support autoregressive rollout —
it just probes what's already in the state.

Method
------
For each horizon k = 1..K, fit a linear regression `state_t @ W_k ≈ x_{t+k}` on
SSv2 train, eval per-horizon L2 on SSv2 val. Crucially, fit a second probe in
parallel: `x_t @ W_k_raw ≈ x_{t+k}` (raw-DINO probe, no encoder). The gap
`raw_probe_err - state_probe_err` is exactly "what the encoder's temporal
modeling contributes at horizon k, beyond linearly projecting the current
frame forward."

All probes are closed-form ridge regression (one pass over train to accumulate
X^T X and X^T Y, one solve, one pass over val to eval). No SGD, no epochs,
no hyperparameters beyond a tiny ridge λ for numerical stability.

Baselines per horizon
---------------------
- Copy              : ||x_t - x_{t+k}||^2                      (zero-param drift floor)
- Raw-DINO probe    : ||x_t @ W_k_raw - x_{t+k}||^2            (linear in current frame only)
- Mean-pool probe   : ||mean(x_0..x_t) @ W_k_mean - x_{t+k}||^2  (unordered multi-frame aggregation)
- History probe     : ||concat_leftpad(x_0..x_t) @ W_k_hist - x_{t+k}||^2
                      (linear in full ordered past, left-padded to length T — strongest linear ceiling)
- State probe       : ||state_t @ W_k_state - x_{t+k}||^2      (linear in encoder state)

Full additive decomposition:
  copy - state = (copy - raw) + (raw - mean) + (mean - hist) + (hist - state)
                  LIN PROJ      AGGREG         LIN DYNAMICS    NONLIN DYNAMICS

- LIN PROJ        : gain from linearly projecting the current frame.
- AGGREG          : gain from seeing multiple frames without ordering (denoising). Can flip
                    sign at short k (mean-pool isn't nested in lin_reg_last).
- LIN DYNAMICS    : gain from linear combinations of *ordered* past frames (velocity,
                    acceleration, learned EMA). hist strictly dominates mean-pool (can set
                    uniform-average weights), so this term is ≥ 0.
- NONLIN DYNAMICS : gain from the encoder's *nonlinear* state update — the only column
                    specific to the RNN being nonlinear. If ≤ 0, a linear AR model matches
                    the RNN; the nonlinearity is unused.

Interpretation
--------------
- state < hist        : RNN's nonlinearity adds value beyond any linear combination of past frames.
- state ≈ hist        : a linear AR model is doing everything the RNN does.
- state < mean-pool   : temporal ordering contributes beyond unordered aggregation.
- Trend across k decides between "myopic", "partial", and "fully dynamic".

Token handling for patch models
-------------------------------
For patches (S=256), the probe is applied token-wise — one shared `W: D×D`
across all 256 spatial positions. Matches `w_pred`'s inductive bias and the
Table 3 / static_dynamic error convention (sum over D, mean over S). Cannot
capture cross-patch motion ("patch (5,8) → (5,9)") — that would need an
(S·D)×(S·D) probe, ~9B params, infeasible. Document the limitation.

Scope
-----
Any RNN checkpoint (learned-space or DINO-space). The linear probe handles the
projection, so `predict_in_dino_space` is not required. Causal transformer
checkpoints could be supported identically (the encoder's `outs` tensor has
the same shape semantics), but not wired up in v1.

Usage
-----
    PYTHONPATH=/home/manu/vjepa2 python root/evals/multi_horizon_probe.py \
        --checkpoint /nas/manu/vjepa2/outputs/<run>/last.pt \
        --data_dir /nas/manu \
        --max_horizon 4 --batch_size 128 --gpu 0
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


def check_supported(model_args):
    enc_type = model_args.encoder.type
    if enc_type != "rnn":
        raise RuntimeError(
            f"multi_horizon_probe currently supports encoder.type='rnn' only (got '{enc_type}'). "
            "Causal transformer support can be added — encoder.outs tensor has the same shape."
        )


def pick_feat_suffix(model_args):
    use_patches = getattr(model_args, "use_patch_tokens", False)
    meanpool_patches = getattr(model_args, "meanpool_patches", False)
    if meanpool_patches:
        return "_meanpool"
    if use_patches:
        return "_patches"
    return ""


def load_split_feats(data_dir, dino_name, suffix, split):
    """Load cached features for a split, truncating the zero-padded CLS val cache."""
    path = os.path.join(data_dir, "ssv2/dino_feats", dino_name, f"{split}{suffix}.pt")
    print(f"Loading {split} features from {path}")
    feats = torch.load(path, mmap=True)
    print(f"  shape: {tuple(feats.shape)}, dtype: {feats.dtype}")

    # Only validation.pt (CLS split) is padded; validation_patches.pt is
    # correctly sized; train splits are correctly sized. Truncate anything
    # that looks oversized relative to the split's CSV row count.
    csv_path = os.path.join(data_dir, f"ssv2/data/{split}.csv")
    if os.path.exists(csv_path):
        n_valid = sum(1 for _ in open(csv_path))
        if feats.shape[0] > n_valid:
            print(f"  truncating padded cache: {feats.shape[0]} -> {n_valid}")
            feats = feats[:n_valid]
    return feats


def init_accumulators(max_k, D, T, device):
    """XtX and XtY accumulators for every probe, per horizon.

    History probe uses left-padded full-T concat of length T*D so one shared W
    is fit across varying t within each k.
    """
    TD = T * D
    accs = {}
    for k in range(1, max_k + 1):
        accs[k] = {
            "XtX_state": torch.zeros(D, D, device=device, dtype=torch.float64),
            "XtY_state": torch.zeros(D, D, device=device, dtype=torch.float64),
            "XtX_raw":   torch.zeros(D, D, device=device, dtype=torch.float64),
            "XtY_raw":   torch.zeros(D, D, device=device, dtype=torch.float64),
            "XtX_mean":  torch.zeros(D, D, device=device, dtype=torch.float64),
            "XtY_mean":  torch.zeros(D, D, device=device, dtype=torch.float64),
            "XtX_hist":  torch.zeros(TD, TD, device=device, dtype=torch.float64),
            "XtY_hist":  torch.zeros(TD, D, device=device, dtype=torch.float64),
            "n": 0,
        }
    return accs


def build_hist_leftpad(x, t, T):
    """Left-pad concat(x_0..x_t) with zeros to fixed time-length T, then flatten.

    Most-recent frame x_t always occupies the last D-slot so a shared W across
    different t sees a consistent 'recency' position. Returns (B_flat, T*D)
    where B_flat = B for CLS or B*S for patches (per-token concat).
    """
    if x.dim() == 3:
        B, _, D = x.shape
        hist = x[:, :t + 1]  # (B, t+1, D)
        pad_len = T - (t + 1)
        if pad_len > 0:
            pad = torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)
            hist = torch.cat([pad, hist], dim=1)  # (B, T, D)
        return hist.reshape(B, T * D)
    else:
        B, _, S, D = x.shape
        hist = x[:, :t + 1]  # (B, t+1, S, D)
        pad_len = T - (t + 1)
        if pad_len > 0:
            pad = torch.zeros(B, pad_len, S, D, device=x.device, dtype=x.dtype)
            hist = torch.cat([pad, hist], dim=1)  # (B, T, S, D)
        # (B, T, S, D) -> (B, S, T, D) -> (B*S, T*D), per-token concat
        return hist.permute(0, 2, 1, 3).reshape(B * S, T * D)


@torch.no_grad()
def accumulate_split(model, feats, accs, max_k, batch_size, device, desc):
    """One pass: run encoder, accumulate XtX and XtY for all horizons and both probes."""
    N, T = feats.shape[0], feats.shape[1]
    D = feats.shape[-1]

    for start in tqdm(range(0, N, batch_size), desc=desc):
        end = min(start + batch_size, N)
        x = feats[start:end].to(device, dtype=torch.float32)  # (B, T, D) or (B, T, S, D)
        outs, *_ = model.encoder(x)  # same shape as x; outs[:, t] = state after frame t

        # Flatten non-(T,D) dims at each t so both CLS (B,D) and patches (B,S,D) work
        # with the same code path. For patches each patch counts as one training row
        # — token-wise probe.
        for k in range(1, max_k + 1):
            for t in range(1, T - k):  # skip t=0 (state initialized from zeros)
                state_t = outs[:, t].reshape(-1, D).to(torch.float64)   # (Bp, D)
                raw_t   = x[:, t].reshape(-1, D).to(torch.float64)
                target  = x[:, t + k].reshape(-1, D).to(torch.float64)
                # mean-pool of frames 0..t — unordered multi-frame aggregation
                mean_t  = x[:, :t + 1].mean(dim=1).reshape(-1, D).to(torch.float64)
                # left-padded concat(x_0..x_t) — ordered linear history ceiling
                hist_t  = build_hist_leftpad(x, t, T).to(torch.float64)

                accs[k]["XtX_state"] += state_t.T @ state_t
                accs[k]["XtY_state"] += state_t.T @ target
                accs[k]["XtX_raw"]   += raw_t.T @ raw_t
                accs[k]["XtY_raw"]   += raw_t.T @ target
                accs[k]["XtX_mean"]  += mean_t.T @ mean_t
                accs[k]["XtY_mean"]  += mean_t.T @ target
                accs[k]["XtX_hist"]  += hist_t.T @ hist_t
                accs[k]["XtY_hist"]  += hist_t.T @ target
                accs[k]["n"]         += state_t.shape[0]


def solve_ridge(XtX, XtY, lam_rel):
    """Closed-form ridge regression with relative regularization.

    W = (XtX + (lam_rel * trace(XtX) / D) * I)^-1 XtY
    """
    D = XtX.shape[0]
    reg = lam_rel * torch.trace(XtX) / D
    eye = torch.eye(D, device=XtX.device, dtype=XtX.dtype)
    return torch.linalg.solve(XtX + reg * eye, XtY)


def fit_probes(accs, lam_rel):
    """Solve ridge for all horizons, every probe. Returns dicts keyed by horizon k."""
    Ws_state, Ws_raw, Ws_mean, Ws_hist = {}, {}, {}, {}
    for k, a in accs.items():
        Ws_state[k] = solve_ridge(a["XtX_state"], a["XtY_state"], lam_rel).to(torch.float32)
        Ws_raw[k]   = solve_ridge(a["XtX_raw"],   a["XtY_raw"],   lam_rel).to(torch.float32)
        Ws_mean[k]  = solve_ridge(a["XtX_mean"],  a["XtY_mean"],  lam_rel).to(torch.float32)
        Ws_hist[k]  = solve_ridge(a["XtX_hist"],  a["XtY_hist"],  lam_rel).to(torch.float32)
    return Ws_state, Ws_raw, Ws_mean, Ws_hist


@torch.no_grad()
def eval_split(model, feats, Ws_state, Ws_raw, Ws_mean, Ws_hist, max_k, batch_size, device):
    """One pass over val: accumulate sum-over-D L2 per horizon for copy / raw / mean / hist / state."""
    N, T = feats.shape[0], feats.shape[1]
    D = feats.shape[-1]

    sums = {k: {"copy": 0.0, "raw": 0.0, "mean": 0.0, "hist": 0.0, "state": 0.0, "n": 0}
            for k in range(1, max_k + 1)}

    for start in tqdm(range(0, N, batch_size), desc="val eval"):
        end = min(start + batch_size, N)
        x = feats[start:end].to(device, dtype=torch.float32)
        outs, *_ = model.encoder(x)

        for k in range(1, max_k + 1):
            for t in range(1, T - k):
                state_t = outs[:, t].reshape(-1, D)   # (Bp, D)
                raw_t   = x[:, t].reshape(-1, D)
                mean_t  = x[:, :t + 1].mean(dim=1).reshape(-1, D)
                hist_t  = build_hist_leftpad(x, t, T)  # (Bp, T*D), fp32
                target  = x[:, t + k].reshape(-1, D)

                pred_state = state_t @ Ws_state[k]
                pred_raw   = raw_t   @ Ws_raw[k]
                pred_mean  = mean_t  @ Ws_mean[k]
                pred_hist  = hist_t  @ Ws_hist[k]

                err_state = ((pred_state - target) ** 2).sum(dim=-1)  # (Bp,)
                err_raw   = ((pred_raw   - target) ** 2).sum(dim=-1)
                err_mean  = ((pred_mean  - target) ** 2).sum(dim=-1)
                err_hist  = ((pred_hist  - target) ** 2).sum(dim=-1)
                err_copy  = ((raw_t      - target) ** 2).sum(dim=-1)

                sums[k]["state"] += err_state.sum().item()
                sums[k]["raw"]   += err_raw.sum().item()
                sums[k]["mean"]  += err_mean.sum().item()
                sums[k]["hist"]  += err_hist.sum().item()
                sums[k]["copy"]  += err_copy.sum().item()
                sums[k]["n"]     += err_state.shape[0]

    return {
        k: {m: sums[k][m] / max(sums[k]["n"], 1)
            for m in ("copy", "raw", "mean", "hist", "state")}
        for k in sums
    }


def print_results(results, n_train_anchors_per_k):
    print()
    print("=" * 128)
    print("MULTI-HORIZON LINEAR PROBE")
    print("=" * 128)
    print(f"{'k':>4s} {'copy':>10s} {'raw':>10s} {'mean-pool':>10s} {'hist':>10s} {'state':>10s} "
          f"{'| TOTAL':>10s} {'LIN PROJ':>10s} {'AGGREG':>10s} {'LIN DYN':>10s} {'NONLIN DYN':>10s}")
    print(f"{'':>4s} {'':>10s} {'':>10s} {'':>10s} {'':>10s} {'':>10s} "
          f"{'| cop-sta':>10s} {'cop-raw':>10s} {'raw-mean':>10s} {'mean-hist':>10s} {'hist-sta':>10s}")
    print("-" * 128)
    for k, row in results.items():
        total      = row["copy"] - row["state"]
        lin_proj   = row["copy"] - row["raw"]
        aggreg     = row["raw"]  - row["mean"]
        lin_dyn    = row["mean"] - row["hist"]
        nonlin_dyn = row["hist"] - row["state"]
        print(f"{k:>4d} {row['copy']:>10.1f} {row['raw']:>10.1f} {row['mean']:>10.1f} "
              f"{row['hist']:>10.1f} {row['state']:>10.1f} "
              f"{'|':>2s}{total:>8.1f} {lin_proj:>10.1f} {aggreg:>10.1f} {lin_dyn:>10.1f} {nonlin_dyn:>10.1f}")
    print()
    # Verdict
    beats_hist = [k for k, r in results.items() if r["state"] < r["hist"]]
    beats_mean = [k for k, r in results.items() if r["state"] < r["mean"]]
    beats_raw  = [k for k, r in results.items() if r["state"] < r["raw"]]
    print(f"Horizons where state < hist (RNN nonlin adds value): {beats_hist or 'none'}")
    print(f"Horizons where state < mean-pool:                    {beats_mean or 'none'}")
    print(f"Horizons where state < raw-DINO:                     {beats_raw or 'none'}")
    if beats_hist:
        print("→ State beats the linear-AR ceiling — RNN's nonlinearity contributes beyond linear dynamics.")
    else:
        print("→ State does NOT beat hist — a linear AR model matches the RNN; nonlinearity unused.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--max_horizon", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--ridge_lambda", type=float, default=1e-3,
                   help="Relative ridge strength: reg = lambda * trace(XtX) / D.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--train_limit", type=int, default=None,
                   help="Optional cap on number of train videos (for quick smoke tests).")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    model, model_args = load_model(args.checkpoint, device)
    check_supported(model_args)

    suffix = pick_feat_suffix(model_args)
    dino_name = model_args.dino_model.split("_")[-1]
    train_feats = load_split_feats(args.data_dir, dino_name, suffix, "train")
    val_feats   = load_split_feats(args.data_dir, dino_name, suffix, "validation")
    if args.train_limit is not None:
        train_feats = train_feats[: args.train_limit]
        print(f"Capped train features to {train_feats.shape[0]}")

    D = train_feats.shape[-1]
    T = train_feats.shape[1]
    if args.max_horizon >= T:
        raise ValueError(f"max_horizon ({args.max_horizon}) must be < T ({T}); need t+k ≤ T-1 with t ≥ 1.")

    accs = init_accumulators(args.max_horizon, D, T, device)

    print("\n[1/3] Accumulating XtX / XtY over train split...")
    accumulate_split(model, train_feats, accs, args.max_horizon, args.batch_size, device, "train")

    print("\n[2/3] Solving ridge regression (closed form)...")
    Ws_state, Ws_raw, Ws_mean, Ws_hist = fit_probes(accs, args.ridge_lambda)
    for k, a in accs.items():
        print(f"  horizon k={k}: fit on {a['n']:,} rows")

    print("\n[3/3] Evaluating on val split...")
    results = eval_split(model, val_feats, Ws_state, Ws_raw, Ws_mean, Ws_hist,
                         args.max_horizon, args.batch_size, device)

    train_n_per_k = {k: accs[k]["n"] for k in accs}
    print_results(results, train_n_per_k)


if __name__ == "__main__":
    main()
