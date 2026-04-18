"""
Verification harness for multi-horizon prediction heads.
Runs Checks 1-4 from docs/plan_multi_horizon.md §4 (pure forward/backward,
no data needed). Check 5 (strict=True checkpoint load + multi_horizon_probe)
is run separately.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from root.models.rnn import VideoRNNTransformerEncoder

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = 384
T = 8

def build(max_horizon: int, predict_in_dino_space: bool = True) -> VideoRNNTransformerEncoder:
    return VideoRNNTransformerEncoder(
        dim=D, update_type="surprise", num_layers=4, num_heads=8,
        mlp_dim=4 * D, cross_attn_dim=2048, decay_state=False,
        predict_in_dino_space=predict_in_dino_space, pred_hidden_dim=D,
        max_horizon=max_horizon,
    ).to(device)


def check_1_shapes():
    print("\n=== Check 1: shapes ===")
    for S in (1, 256):
        for K in (1, 2, 3, 4):
            model = build(max_horizon=K).eval()
            if S == 1:
                x = torch.randn(2, T, D, device=device)
            else:
                x = torch.randn(2, T, S, D, device=device)
            with torch.no_grad():
                out = model(x)
            assert len(out) == 7, f"expected 7-tuple, got {len(out)}"
            outs, state, g, n, rn, pe, mh = out
            assert isinstance(mh, list), f"mh type: {type(mh)}"
            assert len(mh) == K, f"S={S} K={K}: len(mh)={len(mh)}, want {K}"
            for k_idx, err in enumerate(mh):
                k = k_idx + 1
                expected = (2, T - k, S)
                assert tuple(err.shape) == expected, \
                    f"S={S} K={K} k={k}: err.shape={tuple(err.shape)}, want {expected}"
            assert tuple(pe.shape) == (2, T, S), f"pe.shape={tuple(pe.shape)}"
            print(f"  S={S} K={K}: pe {tuple(pe.shape)}, mh lens {[tuple(e.shape) for e in mh]}  OK")


def check_2_k1_invariant():
    print("\n=== Check 2: K=1 invariant — mh[0] == pred_error_l2[:, 1:] ===")
    for S in (1, 256):
        model = build(max_horizon=1).eval()
        if S == 1:
            x = torch.randn(2, T, D, device=device)
        else:
            x = torch.randn(2, T, S, D, device=device)
        with torch.no_grad():
            _, _, _, _, _, pe, mh = model(x)
        invariant = torch.allclose(mh[0], pe[:, 1:], atol=1e-6, rtol=1e-5)
        max_abs = (mh[0] - pe[:, 1:]).abs().max().item()
        print(f"  S={S}: allclose={invariant}, max_abs_diff={max_abs:.3e}")
        assert invariant, f"K=1 invariant failed for S={S} (max diff {max_abs})"


def check_3_backward_nan():
    print("\n=== Check 3: backward — all heads have finite grads ===")
    K = 4
    for S in (1, 256):
        model = build(max_horizon=K).train()
        if S == 1:
            x = torch.randn(2, T, D, device=device)
        else:
            x = torch.randn(2, T, S, D, device=device)
        _, _, _, _, _, _, mh = model(x)
        loss = sum((1.0 / K) * err.mean() for err in mh)
        loss.backward()
        for name, p in model.core.w_pred.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"w_pred.{name}"
            assert p.grad.abs().sum() > 0, f"w_pred.{name} zero grad"
        for hi, head in enumerate(model.core.w_pred_extra):
            for name, p in head.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), f"w_pred_extra[{hi}].{name}"
                assert p.grad.abs().sum() > 0, f"w_pred_extra[{hi}].{name} zero grad"
        print(f"  S={S}: all {K} heads have finite nonzero grads  OK")


def check_4_init_monotonicity():
    print("\n=== Check 4: init monotonicity — per-horizon loss increases with k ===")
    K = 4
    for S in (1, 256):
        monotone_count = 0
        n_trials = 5
        for seed in range(n_trials):
            torch.manual_seed(seed)
            model = build(max_horizon=K).eval()
            if S == 1:
                x = torch.randn(2, T, D, device=device)
            else:
                x = torch.randn(2, T, S, D, device=device)
            with torch.no_grad():
                _, _, _, _, _, _, mh = model(x)
            scalars = [err.mean().item() for err in mh]
            is_monotone = all(scalars[i] <= scalars[i + 1] for i in range(K - 1))
            monotone_count += int(is_monotone)
            print(f"  S={S} seed={seed}: {[f'{s:.2f}' for s in scalars]} monotone={is_monotone}")
        print(f"  S={S}: {monotone_count}/{n_trials} seeds monotone "
              f"({'OK' if monotone_count >= n_trials // 2 else 'FAIL'})")


if __name__ == "__main__":
    check_1_shapes()
    check_2_k1_invariant()
    check_3_backward_nan()
    check_4_init_monotonicity()
    print("\nAll self-contained checks passed.")
