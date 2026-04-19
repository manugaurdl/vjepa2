# Data Layout & Reproducibility

`train.py` consumes Something-Something-v2 (SSv2) with DINOv2 features. All data lives under `{--data_dir}/ssv2/` (default `/nas/manu`). Nothing outside `ssv2/` is read during training.

## Files under `ssv2/`

```
ssv2/
├── 20bn-something-something-v2/    # raw .webm videos (~220k files, ~19 GB)
├── data/
│   ├── train.csv                   # 168,913 rows
│   ├── validation.csv              #  24,777 rows
│   └── test.csv                    # empty (labels not public)
├── labels/
│   ├── labels.json                 # 174 class templates → class_id
│   ├── train.json                  # per-video annotations
│   └── validation.json
└── dino_feats/vits14/              # OPTIONAL pre-extracted features
    ├── train.pt                    # 1.9  GB   CLS token,        [N, 8, 384]
    ├── validation.pt               # 1.9  GB   CLS (padded to 168,913 — see gotcha)
    ├── train_meanpool.pt           # 1.0  GB   mean-pooled patches
    ├── validation_meanpool.pt      # 152  MB
    ├── train_patches.pt            # 265  GB   full patch grid,  [N, 8, 256, 384]
    └── validation_patches.pt       #  39  GB
```

CSV format (space-delimited, no header):
```
/any/prefix/ssv2/20bn-something-something-v2/78687.webm 19
```
The loader chops the prefix at the first `ssv2/` and rejoins with `--data_dir` (`src/datasets/video_dataset.py:213`), so an in-CSV prefix from a different machine still works.

## What `train.py` actually opens

| Flags | Files touched |
|---|---|
| default | `data/{train,validation}.csv` + raw `.webm` (decord decode + live DINO) |
| `--load_cache_feats` | `dino_feats/vits14/{train,validation}.pt` (CLS) |
| `--load_cache_feats --meanpool_patches` | `..._meanpool.pt` |
| `--load_cache_feats --use_patch_tokens` | `..._patches.pt` |
| `--cache_dino_feats` | reads CSVs + raw videos, **writes** the cache files above |
| `--ood_eval_csv <path>` | extra CSV (e.g., UCF101), absolute paths inside |

Cache path template (`src/datasets/video_dataset.py:171`):
```
{data_dir}/ssv2/dino_feats/{dino_model.split('_')[-1]}/{split}{suffix}.pt
# suffix ∈ {"", "_meanpool", "_patches"}
```

## Reproduce on a fresh machine

### Option A — transfer the 19 GB archive (fastest)

On the source machine (already done here):
```bash
tar -cf /nas/manu/ssv2_transfer.tar -C /nas/manu \
    ssv2/20bn-something-something-v2 ssv2/data ssv2/labels
```
The archive is uncompressed on purpose — webm is already compressed, so gzip/zip gains nothing and costs 2–3× wall-time. It excludes `dino_feats/` (rebuild locally) and the original download-package zip.

On the new machine:
```bash
mkdir -p $DATA_DIR
tar -xf ssv2_transfer.tar -C $DATA_DIR
```
CSV paths are portable: the loader chops at `ssv2/` and rejoins with `--data_dir`, so the `/nas/manu/...` prefixes baked into the CSVs still resolve correctly.

### Option B — build from scratch

1. **Download SSv2 videos** from https://www.qualcomm.com/developer/software/something-something-v-2-dataset (free login). Extract into `$DATA_DIR/ssv2/20bn-something-something-v2/`.
2. **Place the official labels** (`labels.json`, `train.json`, `validation.json`) under `$DATA_DIR/ssv2/labels/`.
3. **Generate the CSVs** from the JSONs: for each entry in `train.json`/`validation.json`, write
   ```
   $DATA_DIR/ssv2/20bn-something-something-v2/<id>.webm  labels.json[<template>]
   ```
   to `$DATA_DIR/ssv2/data/{train,validation}.csv`.

### Then, on the new machine

4. **(Optional) Build the DINO cache.** Skips video decoding during training.
   ```bash
   # CLS
   python train.py --config base --data_dir $DATA_DIR --cache_dino_feats --epochs 1
   # patch grid (265 GB!)
   python train.py --config base --data_dir $DATA_DIR --cache_dino_feats --use_patch_tokens --epochs 1
   # meanpool
   python train.py --config base --data_dir $DATA_DIR --cache_dino_feats --meanpool_patches --epochs 1
   ```
   Regardless of which route you took to get the raw data, steps 4+5 are identical.
5. **Train.**
   ```bash
   python train.py --config base \
       --wandb.run_name patch_pred_dino_space \
       --data_dir $DATA_DIR --output_dir $DATA_DIR/vjepa2/outputs \
       --load_cache_feats --use_patch_tokens --no-action_classification \
       --encoder.rnn.predict_in_dino_space \
       --batch_size 8 --val_batch_size 8 --epochs 100
   ```

## Gotcha

`validation.pt` (CLS cache) is zero-padded to 168,913 rows. Truncate to 24,777 before averaging any metric, or the mean deflates ~7×. Fix already applied at the save site in `train.py` and in `temporal_shuffle_test.py`; any new consumer of `validation.pt` must do the same. See CLAUDE.md.
