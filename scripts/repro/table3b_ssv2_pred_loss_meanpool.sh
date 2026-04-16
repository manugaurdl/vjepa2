#!/usr/bin/env bash
# Reproduces Table 3b of docs/april_3_meet.md — next-frame pred_loss in
# MEAN-POOLED DINO patch space (S=1, D=384; patches averaged across the 256
# spatial tokens per frame). Same metric as Table 2/3: ((pred - target)**2)
# .sum(-1).mean over (N, t=1..7).
#
# Rows:
#   1. Copy-last-frame baseline on validation_meanpool.pt (data-only)
#   2. Causal Transformer — tbd (no checkpoint)
#   3. RNN  k5qezvem (meanpool_patches + predict_in_dino_space, last.pt)
#
# Verification that `k5qezvem` actually predicts meanpooled DINO features:
#   - args.meanpool_patches=True       -> dataloader loads validation_meanpool.pt (B,T,384)
#   - args.action_classification=False -> pred-only run
#   - encoder.rnn.predict_in_dino_space=True
#     -> GatedTransformerCore.encoder = nn.Identity()  (rnn.py:167)
#     -> h = inputs = raw meanpool DINO features       (rnn.py:191)
#     -> error = h - w_pred(state)                      (rnn.py:198)
#     -> pred_error_l2 = (error**2).sum(-1)             (rnn.py:199)
#   So the loss is computed directly against raw meanpooled DINO targets.
#
# Published Table 3b: copy baseline 162.8 (RNN and Causal Transformer tbd).
#
# Checkpoint: last.pt (no best.pt — pred-only run predates the best.pt fix
# for non-classification training).
#
# Usage:
#   source /home/manu/vjepa2/.venv/bin/activate
#   bash scripts/repro/table3b_ssv2_pred_loss_meanpool.sh

set -euo pipefail

cd /home/manu/vjepa2
export PYTHONPATH=/home/manu/vjepa2

OUT=outputs/repro_table3b
mkdir -p "$OUT"

RNN_CKPT=/nas/manu/vjepa2/outputs/meanpool_patch_pred_dino_space_k5qezvem/last.pt

# --- Row 3: RNN (GPU 0) ---
CUDA_VISIBLE_DEVICES=0 python root/evals/temporal_shuffle_test.py \
    --checkpoint "$RNN_CKPT" \
    --data_dir /nas/manu --batch_size 64 --num_shuffles 1 --gpu 0 \
    > "$OUT/rnn_k5qezvem.log" 2>&1 &
RNN_PID=$!

# --- Row 1: Copy baseline (data-only, CPU) ---
python - <<'PY' > "$OUT/copy_baseline.log"
import torch
feats = torch.load("/nas/manu/ssv2/dino_feats/vits14/validation_meanpool.pt", mmap=True)
n_valid = sum(1 for _ in open("/nas/manu/ssv2/data/validation.csv"))  # 24777
print(f"Loaded {feats.shape}, truncating to {n_valid}")
feats = feats[:n_valid].float()  # (24777, 8, 384)

# Metric: ((f[t]-f[t-1])**2).sum(-1).mean over (N, t=1..7)
diff = feats[:, 1:] - feats[:, :-1]          # (N, 7, 384)
sq = (diff ** 2).sum(dim=-1)                 # (N, 7)
copy_loss = sq.mean().item()
print(f"Copy-last-frame pred_loss: {copy_loss:.4f}  (over {n_valid} valid samples)")
PY

wait $RNN_PID

echo
echo "=== Table 3b reproduction (SSv2 val, MEANPOOL patches, mean pred_error_l2 over t=1..7) ==="
copy=$(grep -oE 'pred_loss: [0-9.]+' "$OUT/copy_baseline.log" | awk '{print $2}')
rnn=$(grep -E '^Normal order pred_loss:' "$OUT/rnn_k5qezvem.log" | awk '{print $NF}')
printf "  %-42s %s\n" "Copy-last-frame baseline"        "$copy"
printf "  %-42s %s\n" "RNN (k5qezvem)"                  "$rnn"
