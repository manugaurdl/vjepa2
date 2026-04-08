#!/usr/bin/env bash
# Reproduces Table 4 of docs/april_3_meet.md (temporal shuffle test).
#
# For each trained model, run eval twice on SSv2 val:
#   1. Normal frame order   -> mean pred_error_l2 over t=1..7
#   2. Shuffled frame order -> same, averaged over 3 random seeds
# Report normal, shuffled, and shuffled/normal ratio.
#
# The ratio is the headline metric: ratio=1 means the model is order-invariant
# (pure set aggregator); ratio>1 means frame order matters. Caveat: ratio>1
# does NOT distinguish "smart EMA/recency bias" from "learned trajectories" —
# see docs/april_3_meet.md interpretation. This test is a necessary-but-not-
# sufficient check for learned dynamics.
#
# Four rows:
#   zyvsy8gk (CLS, CE+pred, learned space)   — best.pt
#   2ldiw9xk (CLS, pred only, DINO space)    — last.pt
#   e6esmgmu (patches, pred only, DINO space)— last.pt
#   tj9x820q (patches, CE+pred, learned space)— best.pt
#
# Absolute pred_loss is in DIFFERENT UNITS across rows:
#   - DINO-space rows (2ldiw9xk, e6esmgmu): error in raw DINO feature space,
#     comparable to Tables 2/3 and to the data-only copy baseline.
#   - Learned-space rows (zyvsy8gk, tj9x820q): error in a W_enc-projected
#     space; absolute numbers look tiny (2-4) and are NOT comparable to
#     anything except the same model's shuffled number. Copy baseline has
#     no meaning here. Only the ratio is interpretable across rows.
#
# Uses the fixed temporal_shuffle_test.py which truncates the padded CLS cache
# to the real 24777 val rows before evaluating (see Table 2 bug writeup).
#
# Usage:
#   source /home/manu/vjepa2/.venv/bin/activate
#   bash scripts/repro/table4_temporal_shuffle.sh

set -euo pipefail

cd /home/manu/vjepa2
export PYTHONPATH=/home/manu/vjepa2

OUT=outputs/repro_table4
mkdir -p "$OUT"

NUM_SHUFFLES=3

# GPU 0: zyvsy8gk (CLS, learned space)
CUDA_VISIBLE_DEVICES=0 python root/evals/temporal_shuffle_test.py \
    --checkpoint '/nas/manu/vjepa2/outputs/update=w(error)_L2weight1e-1_zyvsy8gk/best.pt' \
    --data_dir /nas/manu --batch_size 64 --num_shuffles $NUM_SHUFFLES --gpu 0 \
    > "$OUT/zyvsy8gk.log" 2>&1 &

# GPU 1: 2ldiw9xk (CLS, DINO space)
CUDA_VISIBLE_DEVICES=1 python root/evals/temporal_shuffle_test.py \
    --checkpoint /nas/manu/vjepa2/outputs/pred_in_dino_space_2ldiw9xk/last.pt \
    --data_dir /nas/manu --batch_size 64 --num_shuffles $NUM_SHUFFLES --gpu 0 \
    > "$OUT/2ldiw9xk.log" 2>&1 &

# GPU 2: e6esmgmu (patches, DINO space)
CUDA_VISIBLE_DEVICES=2 python root/evals/temporal_shuffle_test.py \
    --checkpoint /nas/manu/vjepa2/outputs/patch_pred_dino_space_e6esmgmu/last.pt \
    --data_dir /nas/manu --batch_size 32 --num_shuffles $NUM_SHUFFLES --gpu 0 \
    > "$OUT/e6esmgmu.log" 2>&1 &

# GPU 3: tj9x820q (patches, learned space)
CUDA_VISIBLE_DEVICES=3 python root/evals/temporal_shuffle_test.py \
    --checkpoint /nas/manu/vjepa2/outputs/patch_ce_pred_tj9x820q/best.pt \
    --data_dir /nas/manu --batch_size 32 --num_shuffles $NUM_SHUFFLES --gpu 0 \
    > "$OUT/tj9x820q.log" 2>&1 &

wait

# --- Summary ---
extract() {
    # $1 = log path
    normal=$(grep -E '^Normal order pred_loss:' "$1" | awk '{print $NF}')
    shuffled=$(grep -E '^Shuffled order pred_loss' "$1" | awk '{print $NF}')
    ratio=$(grep -E '^Ratio' "$1" | awk '{print $NF}')
    printf "%s\t%s\t%s" "$normal" "$shuffled" "$ratio"
}

echo
echo "=== Table 4 (SSv2 val, mean pred_error_l2 over t=1..7, normal vs shuffled) ==="
printf "%-55s %10s %10s %10s\n" "Model" "Normal" "Shuffled" "Ratio"
for row in \
    "zyvsy8gk (CLS, CE+pred, learned)|zyvsy8gk" \
    "2ldiw9xk (CLS, pred, DINO)|2ldiw9xk" \
    "e6esmgmu (Patch, pred, DINO)|e6esmgmu" \
    "tj9x820q (Patch, CE+pred, learned)|tj9x820q"; do
    label="${row%%|*}"
    key="${row##*|}"
    read -r n s r <<< "$(extract "$OUT/$key.log")"
    printf "%-55s %10s %10s %10s\n" "$label" "$n" "$s" "$r"
done
