#!/usr/bin/env bash
# Reproduces Table 5 of docs/april_3_meet.md (copy baseline shuffle sensitivity).
#
# Data-only diagnostic (no model). For CLS and patches separately, compute:
#   - Normal copy error   = mean over (N, t=1..7) of ||f[t] - f[t-1]||^2
#   - Shuffled copy error = mean over (N, t=1..7) of ||f[perm(t)] - f[perm(t-1)]||^2,
#                           averaged over 3 random permutations per video
#   - Copy shuffle ratio  = shuffled / normal
#
# Interpretation: high copy shuffle ratio means "consecutive frames are much
# more similar than random pairs" (data is smooth in time). Low ratio means
# random pairs are about as different as consecutive ones (flat temporal
# structure). This is a data property, not a model property. It gives
# context for the model shuffle ratio (Table 4): if the data copy ratio is
# 11x but the model ratio is only 1.2x, the model is barely using any of the
# temporal structure that's there.
#
# Published Table 5:
#   CLS (DINO):     copy ratio 11.2x,  model ratio 1.20x
#   Patches (DINO): copy ratio  1.46x, model ratio 1.31x
#
# Reads:
#   /nas/manu/ssv2/dino_feats/vits14/validation.pt          (168913 rows, padded)
#   /nas/manu/ssv2/dino_feats/vits14/validation_patches.pt  (24777 rows, clean)
# Both are truncated to the real val sample count (row count of validation.csv)
# before averaging.
#
# Usage:
#   source /home/manu/vjepa2/.venv/bin/activate
#   bash scripts/repro/table5_copy_shuffle_ratio.sh

set -euo pipefail

cd /home/manu/vjepa2
export PYTHONPATH=/home/manu/vjepa2

OUT=outputs/repro_table5
mkdir -p "$OUT"

python - <<'PY' | tee "$OUT/table5.log"
import torch

NUM_SHUFFLES = 3
VAL_CSV = "/nas/manu/ssv2/data/validation.csv"
n_valid = sum(1 for _ in open(VAL_CSV))

def copy_normal(feats):
    # feats: (N, T, ..., D). Mean of ||f[t]-f[t-1]||^2 over (N, t=1..7, ...).
    # Chunked to keep RAM bounded for patches.
    total_sum = 0.0
    total_count = 0
    chunk = 256
    N = feats.shape[0]
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        fc = feats[s:e].float()
        diff = fc[:, 1:] - fc[:, :-1]
        sq = (diff ** 2).sum(dim=-1)   # (b, T-1, ...)
        total_sum += sq.sum().item()
        total_count += sq.numel()
    return total_sum / total_count

def copy_shuffled(feats, num_shuffles=3):
    # For each video, permute frames and recompute consecutive-pair error.
    # Equivalent to: E[||f[i] - f[j]||^2] over random (i,j) neighboring-in-perm pairs.
    total_sum = 0.0
    total_count = 0
    chunk = 256
    N, T = feats.shape[0], feats.shape[1]
    for seed in range(num_shuffles):
        g = torch.Generator().manual_seed(seed * 10000)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            fc = feats[s:e].float().clone()
            for i in range(fc.shape[0]):
                perm = torch.randperm(T, generator=g)
                fc[i] = fc[i, perm]
            diff = fc[:, 1:] - fc[:, :-1]
            sq = (diff ** 2).sum(dim=-1)
            total_sum += sq.sum().item()
            total_count += sq.numel()
    return total_sum / total_count

# --- CLS ---
print("Loading CLS features...")
feats_cls = torch.load("/nas/manu/ssv2/dino_feats/vits14/validation.pt", mmap=True)
print(f"  shape before truncate: {feats_cls.shape}")
feats_cls = feats_cls[:n_valid]
print(f"  shape after truncate:  {feats_cls.shape}")

cls_normal = copy_normal(feats_cls)
cls_shuffled = copy_shuffled(feats_cls, NUM_SHUFFLES)
cls_ratio = cls_shuffled / cls_normal
print(f"\nCLS (DINO space):")
print(f"  normal copy err:   {cls_normal:.2f}")
print(f"  shuffled copy err: {cls_shuffled:.2f}  (avg over {NUM_SHUFFLES} seeds)")
print(f"  copy shuffle ratio: {cls_ratio:.2f}x")

# --- Patches ---
print("\nLoading patch features...")
feats_p = torch.load("/nas/manu/ssv2/dino_feats/vits14/validation_patches.pt", mmap=True)
print(f"  shape: {feats_p.shape}")
assert feats_p.shape[0] >= n_valid
feats_p = feats_p[:n_valid]

p_normal = copy_normal(feats_p)
p_shuffled = copy_shuffled(feats_p, NUM_SHUFFLES)
p_ratio = p_shuffled / p_normal
print(f"\nPatches (DINO space):")
print(f"  normal copy err:   {p_normal:.2f}")
print(f"  shuffled copy err: {p_shuffled:.2f}  (avg over {NUM_SHUFFLES} seeds)")
print(f"  copy shuffle ratio: {p_ratio:.2f}x")

print("\n=== Table 5 (copy baseline shuffle sensitivity) ===")
print(f"                     Normal       Shuffled    Copy ratio")
print(f"CLS (DINO)         {cls_normal:10.2f}  {cls_shuffled:12.2f}  {cls_ratio:8.2f}x")
print(f"Patches (DINO)     {p_normal:10.2f}  {p_shuffled:12.2f}  {p_ratio:8.2f}x")
PY
