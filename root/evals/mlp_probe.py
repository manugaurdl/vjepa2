"""
MLP probe (evalID: `mlp_probe`).

Counterpart to `multi_horizon_probe.py` (ridge probe). Same (input, target) pair
construction — fits a small MLP with SGD instead of closed-form ridge. Decides
between Possibility 1 (linear AR is the true ceiling under any probe) and
Possibility 2 (nonlinear structure exists; our RNN fails to capture it) per
`docs/meet2.md` TAKEAWAY SO FAR.

Tier 1 scope
------------
Only two probe inputs: `state` (RNN state_t) and `concat_history`
(left-padded concat of x_0..x_t, same as the ridge `hist` baseline). No
`last`/`meanpool` — those are Tier 2.

Decision rule (at k=4, primary; k=2 secondary)
----------------------------------------------
- `mlp(concat_history)` ≤ ridge(`hist`) within ~1%  → Poss 1 (linear AR is the ceiling).
- `mlp(concat_history)` < ridge(`hist`) by ≥2–3%    → Poss 2 (nonlinear structure exists).
- `mlp(state)` vs ridge(`state`): tells us whether the state itself has
  nonlinear content a linear probe misses.

Sanity check (required): `mlp(concat_history, k=1)` MUST ≤ ridge(hist, k=1).
An MLP is strictly more expressive than a linear map; if it's worse, the MLP
is under-fit — retune LR / hidden / epochs before trusting k≥2 numbers.

Usage
-----
    PYTHONPATH=/home/manu/vjepa2 python root/evals/mlp_probe.py \\
        --checkpoint /nas/manu/vjepa2/outputs/<run>/last.pt \\
        --data_dir /nas/manu \\
        --max_horizon 4 --batch_size 128 \\
        --probe_inputs state concat_history \\
        --mlp_hidden 1024 --mlp_layers 2 \\
        --epochs 30 --lr 1e-3 --weight_decay 1e-4 \\
        --seed 0 --gpu 0
"""

import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from root.evals.multi_horizon_probe import (
    load_model,
    check_supported,
    pick_feat_suffix,
    load_split_feats,
    build_hist_leftpad,
    solve_ridge,
)


class MLPProbe(nn.Module):
    """Residual MLP: linear skip `skip(x)` + nonlinear branch `mlp(x)`.

    The nonlinear branch's final layer is zero-initialized, so at init
    forward(x) = skip(x) — i.e. pure linear regression. SGD first drives
    `skip` toward the ridge-regression solution (the linear baseline), then
    the nonlinear branch learns residual corrections if nonlinear structure
    is extractable from the input.

    This matters because without the skip init, SGD on a plain ReLU MLP can
    plateau well above the closed-form ridge optimum (observed empirically:
    plain MLP stuck ~30% above ridge at k=2..4 after 7 epochs).
    """

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2):
        super().__init__()
        self.skip = nn.Linear(in_dim, out_dim)
        layers = []
        prev = in_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.GELU())
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.nonlinear = nn.Sequential(*layers)
        nn.init.zeros_(self.nonlinear[-1].weight)
        nn.init.zeros_(self.nonlinear[-1].bias)

    def forward(self, x):
        return self.skip(x) + self.nonlinear(x)


@torch.no_grad()
def warmstart_skip_with_ridge(model, train_feats, probes, max_k, batch_size,
                              device, input_keys, ridge_lambda=1e-3):
    """Set each probe's `skip` layer to the closed-form ridge solution.

    Reuses the same pair-construction logic as `multi_horizon_probe.py` but
    only accumulates for the probe inputs actually in use (`state` and
    `concat_history`). After this runs, `probe.forward(x) == ridge_W @ x` at
    init; SGD then refines the nonlinear branch (and can further tune skip).
    """
    N, T = train_feats.shape[0], train_feats.shape[1]
    D = train_feats.shape[-1]

    accs = {}
    for ik in input_keys:
        in_dim = D if ik == "state" else T * D
        for k in range(1, max_k + 1):
            accs[(ik, k)] = {
                "XtX": torch.zeros(in_dim, in_dim, device=device, dtype=torch.float64),
                "XtY": torch.zeros(in_dim, D,      device=device, dtype=torch.float64),
            }

    for start in tqdm(range(0, N, batch_size), desc="ridge-init accum"):
        end = min(start + batch_size, N)
        x = train_feats[start:end].to(device, dtype=torch.float32)
        outs, *_ = model.encoder(x)
        for k in range(1, max_k + 1):
            for t in range(1, T - k):
                target = x[:, t + k].reshape(-1, D).to(torch.float64)
                for ik in input_keys:
                    if ik == "state":
                        inp = outs[:, t].reshape(-1, D).to(torch.float64)
                    else:
                        inp = build_hist_leftpad(x, t, T).to(torch.float64)
                    accs[(ik, k)]["XtX"] += inp.T @ inp
                    accs[(ik, k)]["XtY"] += inp.T @ target

    for (ik, k), a in accs.items():
        W = solve_ridge(a["XtX"], a["XtY"], ridge_lambda).to(torch.float32)  # (in_dim, D)
        # nn.Linear weight shape is (out_dim, in_dim); we want y = x @ W so weight = W.T
        probes[(ik, k)].skip.weight.copy_(W.T.contiguous())
        probes[(ik, k)].skip.bias.zero_()


def build_inputs(x, outs, T, D, t, input_keys):
    """Return a dict mapping input_key -> (B_flat, in_dim) tensor at anchor t.

    For CLS: B_flat = B. For patches: B_flat = B*S (one row per spatial position,
    matching the per-token ridge convention).
    """
    result = {}
    if "state" in input_keys:
        result["state"] = outs[:, t].reshape(-1, D)
    if "concat_history" in input_keys:
        result["concat_history"] = build_hist_leftpad(x, t, T)
    return result


@torch.no_grad()
def eval_probes(model, feats, probes, max_k, batch_size, device, input_keys):
    """Val L2 per probe using the ridge convention (sum over D, mean over rows)."""
    N, T = feats.shape[0], feats.shape[1]
    D = feats.shape[-1]
    sums = {key: {"sum": 0.0, "n": 0} for key in probes}
    for p in probes.values():
        p.eval()

    for start in tqdm(range(0, N, batch_size), desc="val", leave=False):
        end = min(start + batch_size, N)
        x = feats[start:end].to(device, dtype=torch.float32)
        outs, *_ = model.encoder(x)
        for k in range(1, max_k + 1):
            for t in range(1, T - k):
                target = x[:, t + k].reshape(-1, D)
                inputs = build_inputs(x, outs, T, D, t, input_keys)
                for ik in input_keys:
                    key = (ik, k)
                    if key not in probes:
                        continue
                    pred = probes[key](inputs[ik])
                    err = ((pred - target) ** 2).sum(dim=-1)  # (B_flat,)
                    sums[key]["sum"] += err.sum().item()
                    sums[key]["n"] += err.shape[0]

    for p in probes.values():
        p.train()
    return {key: sums[key]["sum"] / max(sums[key]["n"], 1) for key in probes}


def train_probes(model, train_feats, val_feats, probes, optimizers, max_k,
                 batch_size, epochs, device, input_keys, patience=3,
                 initial_val=None):
    """Joint training: one encoder pass per batch updates all (input_key, k) probes.

    Per-probe early stopping: a probe that plateaus for `patience` epochs stops
    training (frozen), but remaining probes keep going. `initial_val` seeds
    the best-so-far tracker (used when skip is warm-started to ridge — we
    don't want SGD drift to make results *worse* than the ridge baseline).
    """
    N, T = train_feats.shape[0], train_feats.shape[1]
    D = train_feats.shape[-1]
    best_val = {key: (initial_val.get(key, float("inf")) if initial_val else float("inf"))
                for key in probes}
    bad_epochs = {key: 0 for key in probes}
    frozen = {key: False for key in probes}

    for epoch in range(1, epochs + 1):
        for p in probes.values():
            p.train()
        running = {key: {"sum": 0.0, "n": 0} for key in probes}

        for start in tqdm(range(0, N, batch_size), desc=f"epoch {epoch}"):
            end = min(start + batch_size, N)
            x = train_feats[start:end].to(device, dtype=torch.float32)
            with torch.no_grad():
                outs, *_ = model.encoder(x)

            # One SGD step per (input_key, k), using rows concatenated across all
            # valid anchors t. This matches the ridge objective (average over t)
            # and avoids the per-t gradient bias that destabilized SGD with
            # per-t updates.
            for k in range(1, max_k + 1):
                targets_list = [x[:, t + k].reshape(-1, D) for t in range(1, T - k)]
                inputs_per_t = [build_inputs(x, outs, T, D, t, input_keys)
                                for t in range(1, T - k)]
                tgt = torch.cat(targets_list, dim=0)  # (sum_t(B_flat), D)
                for ik in input_keys:
                    key = (ik, k)
                    if key not in probes or frozen[key]:
                        continue
                    inp = torch.cat([d[ik] for d in inputs_per_t], dim=0)
                    probe = probes[key]
                    opt = optimizers[key]
                    pred = probe(inp)
                    loss = ((pred - tgt) ** 2).sum(dim=-1).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        running[key]["sum"] += loss.item() * pred.shape[0]
                        running[key]["n"] += pred.shape[0]

        val = eval_probes(model, val_feats, probes, max_k, batch_size, device, input_keys)
        msg = f"[epoch {epoch}] " + "  ".join(
            f"{ik}/k{k}: tr={running[(ik, k)]['sum']/max(running[(ik, k)]['n'], 1):.1f} va={val[(ik, k)]:.1f}"
            + (" FROZEN" if frozen[(ik, k)] else "")
            for ik in input_keys for k in range(1, max_k + 1)
        )
        print(msg)
        for key in probes:
            if frozen[key]:
                continue
            if val[key] < best_val[key]:
                best_val[key] = val[key]
                bad_epochs[key] = 0
            else:
                bad_epochs[key] += 1
                if bad_epochs[key] >= patience:
                    frozen[key] = True
                    print(f"  early stop: {key} plateaued (best val={best_val[key]:.1f})")
        if all(frozen.values()):
            print("All probes early-stopped; exiting training.")
            break
    return best_val


# Ridge reference numbers (from docs/exp_progress.md 2026-04-19, multi_horizon_probe,
# `2ldiw9xk` CLS and `e6esmgmu` patches). Shown alongside MLP results for easy
# visual comparison. For any other checkpoint, the reference is informational only.
RIDGE_REF_CLS = {
    ("state", 1): 524.0, ("state", 2): 730.2, ("state", 3): 860.4, ("state", 4): 936.0,
    ("concat_history", 1): 528.1, ("concat_history", 2): 728.8,
    ("concat_history", 3): 853.7, ("concat_history", 4): 929.9,
}
RIDGE_REF_PATCHES = {
    ("state", 1): 867.0, ("state", 2): 1100.2, ("state", 3): 1220.0, ("state", 4): 1279.8,
    ("concat_history", 1): 868.9, ("concat_history", 2): 1093.8,
    ("concat_history", 3): 1208.5, ("concat_history", 4): 1269.8,
}


def print_results(best_val, max_k, input_keys, use_patches):
    ref = RIDGE_REF_PATCHES if use_patches else RIDGE_REF_CLS
    print()
    print("=" * 96)
    print("MLP PROBE — final val L2 (best-epoch per probe; sum over D, mean over rows)")
    print("=" * 96)
    header = f"{'k':>4s}"
    for ik in input_keys:
        header += f"{'ridge(' + ik + ')':>22s}{'mlp(' + ik + ')':>22s}{'Δ (ridge-mlp)':>18s}"
    print(header)
    print("-" * 96)
    for k in range(1, max_k + 1):
        row = f"{k:>4d}"
        for ik in input_keys:
            rv = ref.get((ik, k), float("nan"))
            mv = best_val[(ik, k)]
            row += f"{rv:>22.1f}{mv:>22.1f}{(rv - mv):>18.1f}"
        print(row)
    print("=" * 96)
    # Sanity gate + decision hint
    hist_key = "concat_history"
    if (hist_key, 1) in best_val:
        rv1 = ref.get((hist_key, 1), float("nan"))
        mv1 = best_val[(hist_key, 1)]
        if rv1 == rv1:
            rel1 = (rv1 - mv1) / rv1 * 100
            print(f"\nSanity (k=1): mlp({hist_key}) is {rel1:+.2f}% vs ridge(hist).")
            if rel1 < -1.0:
                print("  [WARNING] MLP worse than ridge at k=1 — MLP is under-fit.")
                print("            Retune (more epochs / bigger hidden / tune lr) before trusting k≥2.")
                return

    k_decision = max_k
    if (hist_key, k_decision) in best_val:
        rv = ref.get((hist_key, k_decision), float("nan"))
        mv = best_val[(hist_key, k_decision)]
        if rv == rv:
            rel = (rv - mv) / rv * 100
            print(f"At k={k_decision}: mlp({hist_key}) is {rel:+.2f}% vs ridge(hist).")
            if rel >= 2.0:
                print("→ Possibility 2: nonlinear structure in raw frames that ridge misses.")
            elif rel <= 1.0 and rel >= -1.0:
                print("→ Possibility 1: linear AR appears to be the ceiling under any probe.")
            elif rel < -1.0:
                print("→ MLP under-fit at k=4; do not interpret as Poss 1. Retune.")
            else:
                print("→ Ambiguous (1-2% gap); consider Tier 2 / Tier 3 / additional seeds.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--max_horizon", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--probe_inputs", nargs="+", default=["state", "concat_history"],
                   choices=["state", "concat_history"])
    p.add_argument("--mlp_hidden", type=int, default=1024)
    p.add_argument("--mlp_layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--ridge_warmstart", action="store_true", default=True,
                   help="Initialize MLP skip layer with closed-form ridge solution. "
                        "Default on — without it, SGD on the 3072-D concat input plateaus "
                        "well above the ridge ceiling (observed empirically).")
    p.add_argument("--no_ridge_warmstart", action="store_false", dest="ridge_warmstart")
    p.add_argument("--ridge_lambda", type=float, default=1e-3,
                   help="Relative ridge strength for warm-start (reg = lam * trace(XtX) / D).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--train_limit", type=int, default=None,
                   help="Cap train videos for smoke tests.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    model, model_args = load_model(args.checkpoint, device)
    check_supported(model_args)

    suffix = pick_feat_suffix(model_args)
    dino_name = model_args.dino_model.split("_")[-1]
    train_feats = load_split_feats(args.data_dir, dino_name, suffix, "train")
    val_feats = load_split_feats(args.data_dir, dino_name, suffix, "validation")
    if args.train_limit is not None:
        train_feats = train_feats[: args.train_limit]
        print(f"Capped train features to {train_feats.shape[0]}")

    D = train_feats.shape[-1]
    T = train_feats.shape[1]
    if args.max_horizon >= T:
        raise ValueError(f"max_horizon {args.max_horizon} must be < T {T}")

    probes = {}
    optimizers = {}
    for ik in args.probe_inputs:
        in_dim = D if ik == "state" else T * D
        for k in range(1, args.max_horizon + 1):
            probe = MLPProbe(in_dim, args.mlp_hidden, D, args.mlp_layers).to(device)
            probes[(ik, k)] = probe
            optimizers[(ik, k)] = torch.optim.Adam(
                probe.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
    n_params = sum(sum(pp.numel() for pp in p.parameters()) for p in probes.values())
    print(f"\nInitialized {len(probes)} MLP probes: {args.probe_inputs} × k=1..{args.max_horizon}")
    print(f"  arch: {args.mlp_layers}-layer, hidden={args.mlp_hidden}")
    print(f"  total params: {n_params:,}")
    print(f"  optimizer: Adam(lr={args.lr}, wd={args.weight_decay})")
    print(f"  epochs: {args.epochs} (patience {args.patience})")
    print(f"  ridge warm-start: {args.ridge_warmstart} (lambda={args.ridge_lambda})")

    init_val = None
    if args.ridge_warmstart:
        print("\n[0/2] Warm-starting skip layers with closed-form ridge...")
        warmstart_skip_with_ridge(
            model, train_feats, probes, args.max_horizon, args.batch_size,
            device, args.probe_inputs, ridge_lambda=args.ridge_lambda,
        )
        # Freeze the skip layer — it's already at the ridge optimum. Train only
        # the nonlinear branch. This turns the question into: "does adding
        # nonlinearity on top of the ridge solution reduce val L2?" Final MLP
        # is guaranteed ≥ ridge (nonlinear can stay at 0), so no under-fit risk
        # when the answer is Poss 1.
        print("  freezing skip layers (ridge-initialized); training nonlinear branch only.")
        for probe in probes.values():
            for p in probe.skip.parameters():
                p.requires_grad_(False)
        # Rebuild optimizers over trainable params only
        for key in probes:
            optimizers[key] = torch.optim.Adam(
                [p for p in probes[key].parameters() if p.requires_grad],
                lr=args.lr, weight_decay=args.weight_decay,
            )
        init_val = eval_probes(
            model, val_feats, probes, args.max_horizon, args.batch_size,
            device, args.probe_inputs,
        )
        print("  post-warmstart val L2 (should match ridge baselines):")
        for ik in args.probe_inputs:
            for k in range(1, args.max_horizon + 1):
                print(f"    {ik}/k{k}: {init_val[(ik, k)]:.1f}")

    print("\n[1/2] Training MLP probes on train split...")
    best_val = train_probes(
        model, train_feats, val_feats, probes, optimizers,
        args.max_horizon, args.batch_size, args.epochs, device,
        args.probe_inputs, patience=args.patience,
        initial_val=init_val,
    )

    use_patches = getattr(model_args, "use_patch_tokens", False)
    print_results(best_val, args.max_horizon, args.probe_inputs, use_patches)


if __name__ == "__main__":
    main()
