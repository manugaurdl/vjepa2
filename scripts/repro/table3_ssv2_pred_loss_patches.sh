#!/usr/bin/env bash
# Reproduces Table 3 of docs/april_3_meet.md.
#
# What Table 3 measures:
#   Same as Table 2 (mean L2 next-frame pred error on SSv2 val, t=0 skipped),
#   but in DINO PATCH token space (S=256 per frame, D=384) instead of CLS.
#   Metric: (pred_t - dino_patch_t)^2.sum(-1).mean over (N, t=1..7, S=256).
#
# Three rows (all patches, same data):
#   1. Copy-last-frame baseline (data-only)
#   2. Causal Transformer (`r55x2lcn`) — full attention over past patch tokens
#   3. RNN (`e6esmgmu`)               — single hidden-state memory
#
# Cache notes:
#   - Reads validation_patches.pt at /nas/manu/ssv2/dino_feats/vits14/.
#     This file is shape (24777, 8, 256, 384), correctly sized — built by
#     precompute_patch_feats.py, NOT by train.py --cache_dino_feats. So unlike
#     validation.pt (CLS), it does NOT need truncation. The reader still reads
#     n_valid from validation.csv defensively.
#   - File is fp16; cast to float() before differencing.
#   - Total tensor is ~60 GB if materialized once, so the copy baseline is
#     computed in chunks (Welford-style running mean over batches of 256).
#
# Checkpoint policy: both models use last.pt (no best.pt exists for either —
# pred-only runs predate the best.pt fix in train.py:349). These are the
# *fully trained* last.pt files from completed runs.
#
# Each model run takes ~1-2 min. Runs in parallel on GPUs 0-1; copy baseline
# runs on CPU in parallel.
#
# Usage:
#   source /home/manu/vjepa2/.venv/bin/activate
#   bash scripts/repro/table3_ssv2_pred_loss_patches.sh

set -euo pipefail

cd /home/manu/vjepa2
export PYTHONPATH=/home/manu/vjepa2

OUT=outputs/repro_table3
mkdir -p "$OUT"

RNN_CKPT=/nas/manu/vjepa2/outputs/patch_pred_dino_space_e6esmgmu/last.pt
TRANSFORMER_CKPT=/nas/manu/vjepa2/outputs/causal_pred_patches_r55x2lcn/last.pt

# --- Model rows (parallel on GPUs 0-1) ---
CUDA_VISIBLE_DEVICES=0 python root/evals/temporal_shuffle_test.py \
    --checkpoint "$RNN_CKPT" \
    --data_dir /nas/manu --batch_size 32 --num_shuffles 1 --gpu 0 \
    > "$OUT/rnn_e6esmgmu.log" 2>&1 &
RNN_PID=$!

CUDA_VISIBLE_DEVICES=1 python root/evals/temporal_shuffle_test.py \
    --checkpoint "$TRANSFORMER_CKPT" \
    --data_dir /nas/manu --batch_size 32 --num_shuffles 1 --gpu 0 \
    > "$OUT/causal_r55x2lcn.log" 2>&1 &
TRF_PID=$!

# --- Copy-last-frame baseline (CPU, chunked) ---
python - <<'PY' > "$OUT/copy_baseline.log"
import torch
feats = torch.load("/nas/manu/ssv2/dino_feats/vits14/validation_patches.pt", mmap=True)  # (24777, 8, 256, 384) fp16
n_valid = sum(1 for _ in open("/nas/manu/ssv2/data/validation.csv"))  # 24777
assert feats.shape[0] >= n_valid, f"unexpected shape {feats.shape}"
feats = feats[:n_valid]
print(f"Loaded {feats.shape}, computing copy baseline in chunks...")

# Metric: ((feats[:,1:] - feats[:,:-1])**2).sum(-1).mean over (N, T-1, S).
# Computed in chunks of 256 videos to fit in RAM. Sample-weighted running mean.
total_sum = 0.0
total_count = 0
chunk = 256
for s in range(0, n_valid, chunk):
    e = min(s + chunk, n_valid)
    fc = feats[s:e].float()                                  # (b, 8, 256, 384)
    diff = fc[:, 1:] - fc[:, :-1]                            # (b, 7, 256, 384)
    sq = (diff ** 2).sum(dim=-1)                             # (b, 7, 256)
    total_sum += sq.sum().item()
    total_count += sq.numel()
copy_loss = total_sum / total_count
print(f"Copy-last-frame pred_loss: {copy_loss:.4f}  (over {n_valid} valid samples, chunked)")
PY

wait $RNN_PID $TRF_PID

# --- Summary ---
echo
echo "=== Table 3 reproduction (SSv2 val, PATCHES, mean pred_error_l2 over t=1..7, S=256) ==="
copy=$(grep -oE 'pred_loss: [0-9.]+' "$OUT/copy_baseline.log" | awk '{print $2}')
rnn=$(grep -E '^Normal order pred_loss:' "$OUT/rnn_e6esmgmu.log" | awk '{print $NF}')
trf=$(grep -E '^Normal order pred_loss:' "$OUT/causal_r55x2lcn.log" | awk '{print $NF}')
printf "  %-42s %s\n" "Copy-last-frame baseline"            "$copy"
printf "  %-42s %s\n" "Causal Transformer (r55x2lcn)"       "$trf"
printf "  %-42s %s\n" "RNN (e6esmgmu)"                      "$rnn"
