# Mean-pooled Patch Tokens Experiment

Train the same DINO-space prediction model as `e6esmgmu` but with mean-pooled patch features (S=256,D=384 → S=1,D=384) instead of full patch tokens. Fills in Table 3b in `april_3_meet.md`.

**Why**: Mean-pooled patches give copy baseline L2=162.8 (vs CLS=609, patches=1085). Want to know if a model trained on this representation learns useful dynamics, or if the near-zero temporal variation makes it a bad training signal.

---

## Step 1: Precompute mean-pooled features

Load cached patch features, mean-pool across the 256 spatial tokens, save as new cache files.

```bash
source /home/manu/vjepa2/.venv/bin/activate
python -c "
import torch

for split in ['train', 'validation']:
    src = f'/nas/manu/ssv2/dino_feats/vits14/{split}_patches.pt'
    dst = f'/nas/manu/ssv2/dino_feats/vits14/{split}_meanpool.pt'
    feats = torch.load(src, mmap=True)          # (N, T, 256, 384) float16
    pooled = feats.float().mean(dim=2).half()   # (N, T, 384) float16
    torch.save(pooled, dst)
    print(f'{split}: {feats.shape} -> {pooled.shape}, saved to {dst}')
"
```

---

## Step 2: Add `--meanpool_patches` flag

One-line change in `src/datasets/video_dataset.py:165`:

```python
# before
suffix = "_patches" if getattr(args, "use_patch_tokens", False) else ""

# after
if getattr(args, "meanpool_patches", False):
    suffix = "_meanpool"
elif getattr(args, "use_patch_tokens", False):
    suffix = "_patches"
else:
    suffix = ""
```

Also add the argument in `train.py` (or wherever args are parsed):

```python
parser.add_argument("--meanpool_patches", action="store_true", default=False)
```

---

## Step 3: Training run

Same config as `e6esmgmu` but with `--meanpool_patches` instead of `--use_patch_tokens`. The model now receives (B, T, 384) — same shape as CLS — so encoder architecture is identical to `2ldiw9xk`.

```bash
source /home/manu/vjepa2/.venv/bin/activate
PYTHONPATH=/home/manu/vjepa2 python train.py \
    --config base \
    --wandb.run_name meanpool_patch_pred_dino_space \
    --data_dir /nas/manu \
    --output_dir /nas/manu/vjepa2/outputs \
    --load_cache_feats \
    --meanpool_patches \
    --no-action_classification \
    --encoder.rnn.predict_in_dino_space \
    --batch_size 8 \
    --val_batch_size 8 \
    --epochs 100
```

---

## Step 4: Eval

Once trained, run the same evals as `e6esmgmu`:

```bash
# Prediction loss (fills Table 3b)
PYTHONPATH=/home/manu/vjepa2 python root/evals/temporal_shuffle_test.py \
    --checkpoint /nas/manu/vjepa2/outputs/meanpool_patch_pred_dino_space_<run_id>/best.pt \
    --data_dir /nas/manu --batch_size 64 --num_shuffles 3 --gpu 0

# UCF101 linear probe (fills Table 1)
PYTHONPATH=/home/manu/vjepa2 python eval_transfer.py \
    --checkpoint /nas/manu/vjepa2/outputs/meanpool_patch_pred_dino_space_<run_id>/best.pt \
    --mode probe \
    --train_csv /nas/manu/ucf101/data/train.csv \
    --data_csv /nas/manu/ucf101/data/test.csv \
    --num_classes 101 --cache_features --no_aug
```
