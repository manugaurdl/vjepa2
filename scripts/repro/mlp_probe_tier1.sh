#!/usr/bin/env bash
# Tier 1 MLP probe (evalID: mlp_probe).
#
# Decides between Possibility 1 and Possibility 2 per docs/meet2.md TAKEAWAY SO FAR:
#   P1 — linear AR is the ceiling under any probe (MLP ≈ ridge at k=4).
#   P2 — nonlinear structure exists; our RNN doesn't capture it (MLP ≪ ridge at k=4).
#
# Probes `state` and `concat_history` for k=1..4 on the two K=1 baselines:
#   CLS     → `2ldiw9xk`  (pred_in_dino_space_2ldiw9xk/last.pt)
#   Patches → `e6esmgmu`  (patch_pred_dino_space_e6esmgmu/last.pt)
#
# Ridge reference numbers (multi_horizon_probe, docs/exp_progress.md 2026-04-19):
#   CLS     hist  : k1=528.1  k2=728.8  k3=853.7  k4=929.9
#   Patches hist  : k1=868.9  k2=1093.8 k3=1208.5 k4=1269.8
# Sanity gate: mlp(hist, k=1) must be ≤ ridge(hist, k=1). MLP strictly more
# expressive; if not, MLP is under-fit — retune before trusting k≥2.
#
# Usage:
#   source /home/manu/vjepa2/.venv/bin/activate
#   bash scripts/repro/mlp_probe_tier1.sh

set -euo pipefail

cd /home/manu/vjepa2
export PYTHONPATH=/home/manu/vjepa2

OUT=outputs/mlp_probe_tier1
mkdir -p "$OUT"

COMMON=( \
    --data_dir /nas/manu \
    --max_horizon 4 \
    --probe_inputs state concat_history \
    --mlp_hidden 1024 --mlp_layers 2 \
    --epochs 20 --lr 3e-4 --weight_decay 1e-4 \
    --patience 5 --seed 0 \
)

# GPU 0: CLS (2ldiw9xk) — fast (~30 min)
CUDA_VISIBLE_DEVICES=0 python root/evals/mlp_probe.py \
    --checkpoint /nas/manu/vjepa2/outputs/pred_in_dino_space_2ldiw9xk/last.pt \
    --batch_size 128 --gpu 0 \
    "${COMMON[@]}" \
    > "$OUT/2ldiw9xk.log" 2>&1 &
PID_CLS=$!

# GPU 1: Patches (e6esmgmu) — slower (~2 h)
CUDA_VISIBLE_DEVICES=1 python root/evals/mlp_probe.py \
    --checkpoint /nas/manu/vjepa2/outputs/patch_pred_dino_space_e6esmgmu/last.pt \
    --batch_size 32 --gpu 0 \
    "${COMMON[@]}" \
    > "$OUT/e6esmgmu.log" 2>&1 &
PID_PATCH=$!

echo "Launched:"
echo "  CLS     (2ldiw9xk) PID=$PID_CLS  → $OUT/2ldiw9xk.log"
echo "  Patches (e6esmgmu) PID=$PID_PATCH → $OUT/e6esmgmu.log"

wait "$PID_CLS"
echo
echo "=== CLS (2ldiw9xk) ==="
tail -n 30 "$OUT/2ldiw9xk.log"

wait "$PID_PATCH"
echo
echo "=== Patches (e6esmgmu) ==="
tail -n 30 "$OUT/e6esmgmu.log"
