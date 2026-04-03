I am building upon vjepa's repository. The entry point of the codebase is train.py. My codebase utilizes functionalities from vjepa's repo (src/). The code I have written resides insied root/.
I am training a RNN on something something v2 dataset.

## Running scripts
- Always set `PYTHONPATH=/home/manu/vjepa2` when running scripts that import from `root/` (e.g. `PYTHONPATH=/home/manu/vjepa2 python root/evals/temporal_shuffle_test.py ...`)
- For wandb API calls, use `wandb.Api(timeout=60)` — the default 19s timeout is too short and causes `ReadTimeoutError`

## Eval: UCF101 Linear Probe

UCF101 data lives at `/nas/manu/ucf101/data/{train,test}.csv`. Checkpoints at `/nas/manu/vjepa2/outputs/<run_name>/last.pt`.

```bash
# RNN transfer probe
PYTHONPATH=/home/manu/vjepa2 python eval_transfer.py --checkpoint /nas/manu/vjepa2/outputs/<run_name>/last.pt \
    --mode probe --train_csv /nas/manu/ucf101/data/train.csv \
    --data_csv /nas/manu/ucf101/data/test.csv \
    --num_classes 101 --cache_features --no_aug --output_dir outputs/eval_ucf101_<run_name>

# DINO mean-pool baseline
PYTHONPATH=/home/manu/vjepa2 python eval_transfer.py --checkpoint /nas/manu/vjepa2/outputs/<run_name>/last.pt \
    --mode probe --baseline --train_csv /nas/manu/ucf101/data/train.csv \
    --data_csv /nas/manu/ucf101/data/test.csv \
    --num_classes 101 --cache_features --no_aug --output_dir outputs/eval_ucf101_baseline

# DINO concat baseline
PYTHONPATH=/home/manu/vjepa2 python eval_transfer.py --checkpoint /nas/manu/vjepa2/outputs/<run_name>/last.pt \
    --mode probe --concat --train_csv /nas/manu/ucf101/data/train.csv \
    --data_csv /nas/manu/ucf101/data/test.csv \
    --num_classes 101 --cache_features --no_aug --output_dir outputs/eval_ucf101_baseline_concat
```