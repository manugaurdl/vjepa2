#!/usr/bin/env bash
# Reproduces Table 2 of docs/april_3_meet.md.
#
# What Table 2 measures:
#   Mean L2 next-frame prediction error on SSv2 validation, in frozen DINO CLS space.
#   Metric: mean over (N, t=1..7) of (pred_t - dino_feat_t)^2.sum(-1).
#   t=0 is skipped because the model has no memory yet at the first frame.
#
# Three rows (all CLS DINO tokens, same data):
#   1. Copy-last-frame baseline (data-only, no model)
#   2. Causal Transformer (`ud2ncxlq`) — explicit memory over all past frames
#   3. RNN          (`2ldiw9xk`)        — single hidden-state memory
#
# Checkpoint policy: each training run saves best.pt + last.pt. We use best.pt by
# default, but for both Table 2 models only last.pt exists on disk (these are the
# *fully trained* last.pt files from completed runs — distinct from the earlier
# bug where last.pt came from an unfinished run, see CLAUDE.md).
#
# Each model run takes ~1 min (forward pass over 24777 cached val features).
# Runs in parallel on GPUs 0-1; copy baseline is computed inline.
#
# Usage:
#   source /home/manu/vjepa2/.venv/bin/activate
#   bash scripts/repro/table2_ssv2_pred_loss_cls.sh

set -euo pipefail

cd /home/manu/vjepa2
export PYTHONPATH=/home/manu/vjepa2

OUT=outputs/repro_table2
mkdir -p "$OUT"

RNN_CKPT=/nas/manu/vjepa2/outputs/pred_in_dino_space_2ldiw9xk/last.pt
TRANSFORMER_CKPT=/nas/manu/vjepa2/outputs/causal_pred_cls_ud2ncxlq/last.pt

# --- Model rows (parallel on GPUs 0-1) ---
# We only need the "Normal order pred_loss" line, but temporal_shuffle_test.py also
# prints the shuffled metric (cheap, useful for Table 4).
CUDA_VISIBLE_DEVICES=0 python root/evals/temporal_shuffle_test.py \
    --checkpoint "$RNN_CKPT" \
    --data_dir /nas/manu --batch_size 64 --num_shuffles 1 --gpu 0 \
    > "$OUT/rnn_2ldiw9xk.log" 2>&1 &
RNN_PID=$!

CUDA_VISIBLE_DEVICES=1 python root/evals/temporal_shuffle_test.py \
    --checkpoint "$TRANSFORMER_CKPT" \
    --data_dir /nas/manu --batch_size 64 --num_shuffles 1 --gpu 0 \
    > "$OUT/causal_ud2ncxlq.log" 2>&1 &
TRF_PID=$!

# --- Copy-last-frame baseline (no GPU) ---
# IMPORTANT: validation.pt is zero-padded to 168913 (the train-split size) due to
# a bug in root/models/model.py:74 — the cache buffer is sized for train regardless
# of split. Always truncate to the real val sample count (= row count of
# validation.csv, currently 24777) before averaging, otherwise the ~144k zero rows
# dilute the mean by ~7x. temporal_shuffle_test.py now truncates internally too.
python - <<'PY' > "$OUT/copy_baseline.log"
import torch
feats = torch.load("/nas/manu/ssv2/dino_feats/vits14/validation.pt", mmap=True)  # (168913, 8, 384) -- padded
n_valid = sum(1 for _ in open("/nas/manu/ssv2/data/validation.csv"))             # 24777
feats = feats[:n_valid].float()
# Predict frame[t] = frame[t-1]; metric = (diff)^2.sum(-1).mean over (N, t=1..7).
# Matches GatedTransformerCore's pred_error_l2 = (h - w_pred(state))^2.sum(dim=-1).
diffs = feats[:, 1:] - feats[:, :-1]
copy_loss = (diffs ** 2).sum(dim=-1).mean().item()
print(f"Copy-last-frame pred_loss: {copy_loss:.4f}  (over {n_valid} valid samples)")
PY

wait $RNN_PID $TRF_PID

# --- Summary ---
echo
echo "=== Table 2 reproduction (SSv2 val, CLS, mean pred_error_l2 over t=1..7) ==="
copy=$(grep -oE 'pred_loss: [0-9.]+' "$OUT/copy_baseline.log" | awk '{print $2}')
rnn=$(grep -E '^Normal order pred_loss:' "$OUT/rnn_2ldiw9xk.log" | awk '{print $NF}')
trf=$(grep -E '^Normal order pred_loss:' "$OUT/causal_ud2ncxlq.log" | awk '{print $NF}')
printf "  %-42s %s\n" "Copy-last-frame baseline"           "$copy"
printf "  %-42s %s\n" "Causal Transformer (ud2ncxlq)"      "$trf"
printf "  %-42s %s\n" "RNN (2ldiw9xk)"                     "$rnn"
