#!/usr/bin/env bash
# Reproduces Table 1 of docs/april_3_meet.md (UCF101 linear probe).
#
# Launches all 6 rows in parallel, one per GPU (uses GPUs 0-5).
# Each row: 20 epochs, AdamW lr=1e-3, batch=64, no aug, features cached after epoch 0.
# Total wall time on free RTX 6000 Ada: ~5 min.
#
# Results land in outputs/repro_table1/<name>/probe_results.json — read best_acc.
# Logs land in outputs/repro_table1/<name>.log
#
# Per-row checkpoint mapping (must use exactly these — best.pt vs last.pt matters,
# see CLAUDE.md and docs/repo_context.md):
#
#   DINO mean-pool          : baseline mode, --checkpoint is a placeholder
#   DINO concat             : baseline mode, --checkpoint is a placeholder
#   2ldiw9xk (CLS, dino)    : pred_in_dino_space_2ldiw9xk/last.pt        (no best.pt exists)
#   zyvsy8gk (CLS, learned) : update=w(error)_L2weight1e-1_zyvsy8gk/best.pt
#   e6esmgmu (Patch, dino)  : patch_pred_dino_space_e6esmgmu/last.pt     (no best.pt exists)
#   tj9x820q (Patch, learn) : patch_ce_pred_tj9x820q/best.pt
#
# Usage:
#   source /home/manu/vjepa2/.venv/bin/activate
#   bash scripts/repro/table1_ucf101_probe.sh

set -euo pipefail

cd /home/manu/vjepa2
export PYTHONPATH=/home/manu/vjepa2

OUT=outputs/repro_table1
mkdir -p "$OUT"

TRAIN_CSV=/nas/manu/ucf101/data/train.csv
TEST_CSV=/nas/manu/ucf101/data/test.csv

# Placeholder checkpoint for baseline modes (--checkpoint is required by argparse but
# its contents are unused when --baseline or --concat is set).
PLACEHOLDER_CKPT='/nas/manu/vjepa2/outputs/update=w(error)_L2weight1e-1_zyvsy8gk/best.pt'

common_args=(
    --mode probe
    --train_csv "$TRAIN_CSV"
    --data_csv "$TEST_CSV"
    --num_classes 101
    --cache_features
    --no_aug
)

# GPU 0: DINO mean-pool baseline (Table 1: 88.0%)
CUDA_VISIBLE_DEVICES=0 python eval_transfer.py \
    --checkpoint "$PLACEHOLDER_CKPT" --baseline \
    "${common_args[@]}" \
    --output_dir "$OUT/dino_meanpool" > "$OUT/dino_meanpool.log" 2>&1 &

# GPU 1: DINO concat baseline (Table 1: 86.0%)
CUDA_VISIBLE_DEVICES=1 python eval_transfer.py \
    --checkpoint "$PLACEHOLDER_CKPT" --concat \
    "${common_args[@]}" \
    --output_dir "$OUT/dino_concat" > "$OUT/dino_concat.log" 2>&1 &

# GPU 2: 2ldiw9xk — RNN CLS, pred only, DINO space (Table 1: 85.4%)
CUDA_VISIBLE_DEVICES=2 python eval_transfer.py \
    --checkpoint /nas/manu/vjepa2/outputs/pred_in_dino_space_2ldiw9xk/last.pt \
    "${common_args[@]}" \
    --output_dir "$OUT/rnn_2ldiw9xk" > "$OUT/rnn_2ldiw9xk.log" 2>&1 &

# GPU 3: zyvsy8gk — RNN CLS, CE+pred, learned space (Table 1: 84.0%)
CUDA_VISIBLE_DEVICES=3 python eval_transfer.py \
    --checkpoint '/nas/manu/vjepa2/outputs/update=w(error)_L2weight1e-1_zyvsy8gk/best.pt' \
    "${common_args[@]}" \
    --output_dir "$OUT/rnn_zyvsy8gk" > "$OUT/rnn_zyvsy8gk.log" 2>&1 &

# GPU 4: e6esmgmu — RNN Patches, pred only, DINO space (Table 1: 81.7%)
CUDA_VISIBLE_DEVICES=4 python eval_transfer.py \
    --checkpoint /nas/manu/vjepa2/outputs/patch_pred_dino_space_e6esmgmu/last.pt \
    "${common_args[@]}" \
    --output_dir "$OUT/rnn_e6esmgmu" > "$OUT/rnn_e6esmgmu.log" 2>&1 &

# GPU 5: tj9x820q — RNN Patches, CE+pred, learned space (Table 1: 78.3%)
CUDA_VISIBLE_DEVICES=5 python eval_transfer.py \
    --checkpoint /nas/manu/vjepa2/outputs/patch_ce_pred_tj9x820q/best.pt \
    "${common_args[@]}" \
    --output_dir "$OUT/rnn_tj9x820q" > "$OUT/rnn_tj9x820q.log" 2>&1 &

wait

echo
echo "=== Table 1 reproduction (best_acc per row) ==="
for d in dino_meanpool dino_concat rnn_2ldiw9xk rnn_zyvsy8gk rnn_e6esmgmu rnn_tj9x820q; do
    acc=$(python -c "import json; print(f\"{json.load(open('$OUT/$d/probe_results.json'))['best_acc']*100:.2f}%\")")
    printf "  %-18s %s\n" "$d" "$acc"
done
