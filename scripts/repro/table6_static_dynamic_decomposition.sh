#!/usr/bin/env bash
# Reproduces Table 6 of docs/april_3_meet.md (static vs dynamic patch decomposition).
#
# For each patch-token model, split the 256 patches per video into:
#   - dynamic = top 50% by per-patch motion score (||f[t]-f[t-1]||^2 avg over t)
#   - static  = bottom 50%
# Then compare copy-last-frame baseline vs model prediction error *separately*
# for each group. Dynamics-aware model → improvement concentrated on dynamic
# patches. Smoothness-only model → equal improvement on both.
#
# Published Table 6 (e6esmgmu, pred only, patches, DINO space):
#   Dynamic patches: copy 1430.6, model 1077.0, improvement 24.7%
#   Static patches:  copy  746.9, model  625.9, improvement 16.2%
#
# Also reports `tj9x820q` (CE + pred, patches, learned space): model error
# dynamic=4.2, static=3.0 (no copy baseline in latent space).
#
# Uses the padded-cache-safe reader in static_dynamic_decomposition.py
# (truncates to 24777 real val rows).
#
# Usage:
#   source /home/manu/vjepa2/.venv/bin/activate
#   bash scripts/repro/table6_static_dynamic_decomposition.sh

set -euo pipefail

cd /home/manu/vjepa2
export PYTHONPATH=/home/manu/vjepa2

OUT=outputs/repro_table6
mkdir -p "$OUT"

# GPU 0: e6esmgmu (pred only, patches, DINO space) — primary row
CUDA_VISIBLE_DEVICES=0 python root/evals/static_dynamic_decomposition.py \
    --checkpoint /nas/manu/vjepa2/outputs/patch_pred_dino_space_e6esmgmu/last.pt \
    --data_dir /nas/manu --batch_size 32 --gpu 0 \
    > "$OUT/e6esmgmu.log" 2>&1 &

# GPU 1: tj9x820q (CE + pred, patches, learned space)
CUDA_VISIBLE_DEVICES=1 python root/evals/static_dynamic_decomposition.py \
    --checkpoint /nas/manu/vjepa2/outputs/patch_ce_pred_tj9x820q/best.pt \
    --data_dir /nas/manu --batch_size 32 --gpu 0 \
    > "$OUT/tj9x820q.log" 2>&1 &

wait

echo
echo "=== Table 6 reproduction ==="
echo
echo "--- e6esmgmu (pred only, patches, DINO space) ---"
grep -E 'Dynamic patches|Static patches|Shuffle ratio|^->' "$OUT/e6esmgmu.log" || true
echo
echo "--- tj9x820q (CE + pred, patches, learned space) ---"
grep -E 'Dynamic patches|Static patches|Shuffle ratio|^->' "$OUT/tj9x820q.log" || true
