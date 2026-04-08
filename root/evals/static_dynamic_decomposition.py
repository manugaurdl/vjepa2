"""
Static vs Dynamic patch decomposition diagnostic.

Splits patch tokens into static (bottom 50% motion) and dynamic (top 50% motion),
then compares model prediction error vs copy baseline separately for each group.

- Dynamics-aware model: improvement concentrated on dynamic patches.
- Smoothness-only model: equal improvement on both.

Also computes copy baseline shuffle ratio on dynamic patches only, to test whether
CLS's high shuffle ratio (11.2x) is explained by CLS tracking dynamic content.

Usage:
    PYTHONPATH=/home/manu/vjepa2 python root/evals/static_dynamic_decomposition.py \
        --checkpoint /nas/manu/vjepa2/outputs/<run>/best.pt \
        --data_dir /nas/manu \
        --batch_size 64 --gpu 0
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


def compute_motion_scores(feats):
    """
    Compute per-patch motion score across a dataset of cached features.

    Args:
        feats: (N, T, S, D) cached DINO patch features

    Returns:
        motion: (N, S) average frame-to-frame L2 per patch per video
    """
    # ||x[t+1, s] - x[t, s]||^2 averaged over t
    diffs = feats[:, 1:] - feats[:, :-1]  # (N, T-1, S, D)
    motion = (diffs ** 2).sum(dim=-1).mean(dim=1)  # (N, S)
    return motion


def compute_motion_mask(motion, dynamic_pct=50):
    """
    Per-video percentile split into static/dynamic patches.

    Args:
        motion: (N, S)
        dynamic_pct: top X% of patches by motion are "dynamic" (default 50 = median)

    Returns:
        dynamic_mask: (N, S) bool — True for dynamic patches
    """
    threshold_pct = 100 - dynamic_pct
    threshold = torch.quantile(motion.float(), threshold_pct / 100.0, dim=1, keepdim=True)  # (N, 1)
    return motion > threshold


def compute_copy_baseline_error(feats, dynamic_mask):
    """
    Copy baseline: predict x_t = x_{t-1}. Compute error separately for static/dynamic.

    Args:
        feats: (N, T, S, D)
        dynamic_mask: (N, S) bool

    Returns:
        copy_err_dynamic, copy_err_static: scalar mean L2 errors
    """
    # (N, T-1, S, D) -> sum over D -> (N, T-1, S)
    copy_err = ((feats[:, 1:] - feats[:, :-1]) ** 2).sum(dim=-1)  # (N, T-1, S)

    # Expand mask to (N, 1, S) for broadcasting over T-1
    mask = dynamic_mask.unsqueeze(1).expand_as(copy_err)

    copy_err_dynamic = copy_err[mask].mean().item()
    copy_err_static = copy_err[~mask].mean().item()
    return copy_err_dynamic, copy_err_static


def compute_copy_shuffle_ratio_dynamic(feats, dynamic_mask, num_shuffles=3):
    """
    Compute copy baseline shuffle ratio on dynamic patches only.

    Normal copy error on dynamic patches vs shuffled copy error on dynamic patches.
    If this ratio is high (close to CLS's 11.2x), it confirms CLS tracks dynamic content.
    """
    # Normal copy error on dynamic patches
    copy_err = ((feats[:, 1:] - feats[:, :-1]) ** 2).sum(dim=-1)  # (N, T-1, S)
    mask = dynamic_mask.unsqueeze(1).expand_as(copy_err)
    normal_err = copy_err[mask].mean().item()

    # Shuffled copy error on dynamic patches (average over seeds)
    shuffle_errs = []
    for seed in range(num_shuffles):
        gen = torch.Generator().manual_seed(seed * 10000)
        feats_shuffled = feats.clone()
        T = feats.shape[1]
        for i in range(feats.shape[0]):
            perm = torch.randperm(T, generator=gen)
            feats_shuffled[i] = feats_shuffled[i, perm]
        shuf_copy_err = ((feats_shuffled[:, 1:] - feats_shuffled[:, :-1]) ** 2).sum(dim=-1)
        shuffle_errs.append(shuf_copy_err[mask].mean().item())

    shuffled_err = sum(shuffle_errs) / len(shuffle_errs)
    return normal_err, shuffled_err, shuffled_err / max(normal_err, 1e-8)


def get_pred_error(model):
    pred_error_l2 = getattr(model, "pred_error_l2", None)
    if pred_error_l2 is None and hasattr(model, "module"):
        pred_error_l2 = getattr(model.module, "pred_error_l2", None)
    return pred_error_l2


@torch.no_grad()
def compute_model_error(model, feats, dynamic_mask, batch_size, device):
    """
    Run model forward pass, collect per-patch pred_error_l2, split by static/dynamic.

    pred_error_l2 shape from model: (B, T, S) — already per-patch.
    We skip t=0 (state initialized to zeros).

    Returns:
        model_err_dynamic, model_err_static: scalar mean L2 errors
    """
    N = feats.shape[0]
    all_dynamic_errs = []
    all_static_errs = []

    for start in tqdm(range(0, N, batch_size), desc="model forward"):
        end = min(start + batch_size, N)
        x = feats[start:end].to(device, dtype=torch.float32)  # (B, T, S, D)
        ds_index = torch.arange(start, end)
        _ = model(x, ds_index)

        pred_error_l2 = get_pred_error(model)
        if pred_error_l2 is None:
            raise RuntimeError("Model did not produce pred_error_l2. Is update_type='surprise'?")

        # pred_error_l2: (B, T, S) — skip t=0
        err = pred_error_l2[:, 1:].cpu()  # (B, T-1, S)
        mask = dynamic_mask[start:end].unsqueeze(1).expand_as(err)  # (B, T-1, S)

        all_dynamic_errs.append(err[mask])
        all_static_errs.append(err[~mask])

    model_err_dynamic = torch.cat(all_dynamic_errs).mean().item()
    model_err_static = torch.cat(all_static_errs).mean().item()
    return model_err_dynamic, model_err_static


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dynamic_pct", type=int, default=50, help="Top X%% of patches by motion are 'dynamic'")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    model, model_args = load_model(args.checkpoint, device)

    use_patches = getattr(model_args, "use_patch_tokens", False)
    if not use_patches:
        print("ERROR: This diagnostic is for patch token models only (use_patch_tokens=True).")
        return

    # Load cached val features
    dino_name = model_args.dino_model.split("_")[-1]
    feat_path = os.path.join(args.data_dir, "ssv2/dino_feats", dino_name, "validation_patches.pt")
    print(f"Loading features from {feat_path}")
    feats = torch.load(feat_path, mmap=True)
    print(f"Feature shape: {feats.shape}, dtype: {feats.dtype}")  # (N, T, S, D)

    # Handle zero-padded validation data
    N_valid = 24777
    feats = feats[:N_valid]
    print(f"Using {N_valid} valid samples")

    # Step 1: Compute motion scores and mask
    print("\nComputing motion scores...")
    motion = compute_motion_scores(feats.float())  # (N, S)
    dynamic_mask = compute_motion_mask(motion, dynamic_pct=args.dynamic_pct)  # (N, S) bool
    n_dynamic = dynamic_mask.float().mean().item()
    print(f"Dynamic patch fraction: {n_dynamic:.2%} (should be ~50%)")

    # Step 2: Copy baseline error per group
    print("\nComputing copy baseline errors...")
    copy_err_dynamic, copy_err_static = compute_copy_baseline_error(feats.float(), dynamic_mask)
    print(f"  Copy baseline (dynamic patches): {copy_err_dynamic:.1f}")
    print(f"  Copy baseline (static patches):  {copy_err_static:.1f}")

    # Step 3: Model prediction error per group
    print("\nComputing model prediction errors...")
    model_err_dynamic, model_err_static = compute_model_error(
        model, feats, dynamic_mask, args.batch_size, device
    )
    print(f"  Model error (dynamic patches): {model_err_dynamic:.1f}")
    print(f"  Model error (static patches):  {model_err_static:.1f}")

    # Step 4: Compare improvement
    improve_dynamic = (copy_err_dynamic - model_err_dynamic) / copy_err_dynamic * 100
    improve_static = (copy_err_static - model_err_static) / copy_err_static * 100

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"{'':30s} {'Copy':>10s} {'Model':>10s} {'Improve':>10s}")
    print(f"{'Dynamic patches':30s} {copy_err_dynamic:10.1f} {model_err_dynamic:10.1f} {improve_dynamic:9.1f}%")
    print(f"{'Static patches':30s} {copy_err_static:10.1f} {model_err_static:10.1f} {improve_static:9.1f}%")
    print()

    if improve_dynamic > improve_static * 1.5:
        print("-> Improvement concentrated on dynamic patches: model learned dynamics.")
    elif improve_dynamic > improve_static * 1.1:
        print("-> Slight bias toward dynamic patches, but not conclusive.")
    else:
        print("-> Equal improvement on both: model is just smoothing, not learning dynamics.")

    # Step 5: Copy shuffle ratio on dynamic patches only
    print(f"\n{'='*60}")
    print(f"COPY SHUFFLE RATIO (dynamic patches only)")
    print(f"{'='*60}")
    normal, shuffled, ratio = compute_copy_shuffle_ratio_dynamic(feats.float(), dynamic_mask)
    print(f"  Normal copy error (dynamic):   {normal:.1f}")
    print(f"  Shuffled copy error (dynamic): {shuffled:.1f}")
    print(f"  Shuffle ratio (dynamic only):  {ratio:.2f}x")
    print(f"  (Compare: all patches = 1.46x, CLS = 11.2x)")
    if ratio > 5.0:
        print("-> Dynamic patch shuffle ratio is high — CLS's 11.2x likely explained by tracking dynamic content.")
    else:
        print("-> Dynamic patch shuffle ratio is moderate — CLS captures something beyond just dynamic patches.")


if __name__ == "__main__":
    main()
