"""
Temporal shuffle test: does frame ordering matter for prediction quality?

Runs eval on SSv2 val with normal and shuffled frame order.
If pred_loss is similar → model ignores temporal structure.
If pred_loss is higher under shuffling → temporal order matters for prediction.
"""

import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from root.models.model import _build_model
from root.utils import dict_to_namespace


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = dict_to_namespace(ckpt["args"])
    # Set fields needed by _build_model but not relevant here
    args.cache_dino_feats = False
    args.load_cache_feats = True
    args.val_dataset_len = None
    model = _build_model(args, device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, args


def get_pred_error(model):
    pred_error_l2 = getattr(model, "pred_error_l2", None)
    if pred_error_l2 is None and hasattr(model, "module"):
        pred_error_l2 = getattr(model.module, "pred_error_l2", None)
    return pred_error_l2


@torch.no_grad()
def eval_pred_loss(model, feats, labels, batch_size, device, shuffle_frames=False, seed=None):
    """Run forward pass over all features, return mean pred_loss."""
    N = feats.shape[0]
    total_loss = 0.0
    total_samples = 0

    for start in tqdm(range(0, N, batch_size), desc="shuffled" if shuffle_frames else "normal"):
        end = min(start + batch_size, N)
        x = feats[start:end].to(device, dtype=torch.float32)  # (B, T, ..., D)

        if shuffle_frames:
            T = x.shape[1]
            if seed is not None:
                gen = torch.Generator().manual_seed(seed + start)
            else:
                gen = None
            for i in range(x.shape[0]):
                perm = torch.randperm(T, generator=gen)
                x[i] = x[i, perm]

        ds_index = torch.arange(start, end)
        _ = model(x, ds_index)

        pred_error_l2 = get_pred_error(model)
        if pred_error_l2 is not None and pred_error_l2.size(1) > 1:
            # Mean over timesteps 1+ (skip t=0) and all other dims
            loss = pred_error_l2[:, 1:].mean().item()
            total_loss += loss * x.shape[0]
            total_samples += x.shape[0]

    return total_loss / max(total_samples, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_shuffles", type=int, default=5, help="Number of random shuffles to average over")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    model, model_args = load_model(args.checkpoint, device)

    # Load cached val features
    use_patches = getattr(model_args, "use_patch_tokens", False)
    suffix = "_patches" if use_patches else ""
    dino_name = model_args.dino_model.split("_")[-1]
    feat_path = os.path.join(args.data_dir, "ssv2/dino_feats", dino_name, f"validation{suffix}.pt")
    print(f"Loading features from {feat_path}")
    feats = torch.load(feat_path, mmap=True)
    print(f"Feature shape: {feats.shape}, dtype: {feats.dtype}")

    # validation.pt (CLS) is zero-padded to 168913 (size of the train split) — see
    # root/models/model.py:74 for the cause. Truncate to the real number of val
    # samples (= row count of validation.csv, currently 24777). validation_patches.pt
    # is correctly sized by precompute_patch_feats.py, but truncating is harmless.
    val_csv = os.path.join(args.data_dir, "ssv2/data/validation.csv")
    n_valid = sum(1 for _ in open(val_csv))
    if feats.shape[0] > n_valid:
        print(f"Truncating padded cache: {feats.shape[0]} -> {n_valid} (real val samples)")
        feats = feats[:n_valid]

    # Dummy labels (not used for pred_loss)
    labels = torch.zeros(feats.shape[0], dtype=torch.long)

    # Normal eval
    normal_loss = eval_pred_loss(model, feats, labels, args.batch_size, device, shuffle_frames=False)
    print(f"\nNormal order pred_loss:  {normal_loss:.4f}")

    # Shuffled eval (average over multiple seeds)
    shuffle_losses = []
    for seed in range(args.num_shuffles):
        loss = eval_pred_loss(model, feats, labels, args.batch_size, device, shuffle_frames=True, seed=seed * 10000)
        shuffle_losses.append(loss)
        print(f"  Shuffle seed {seed}: {loss:.4f}")

    mean_shuffle_loss = sum(shuffle_losses) / len(shuffle_losses)
    print(f"\nShuffled order pred_loss (avg over {args.num_shuffles} seeds): {mean_shuffle_loss:.4f}")
    print(f"Ratio (shuffled / normal): {mean_shuffle_loss / max(normal_loss, 1e-8):.4f}")
    print(f"Delta: {mean_shuffle_loss - normal_loss:.4f}")

    if mean_shuffle_loss > normal_loss * 1.05:
        print("\n→ Temporal order matters for prediction (>5% increase under shuffling)")
    else:
        print("\n→ Temporal order has minimal effect on prediction quality")


if __name__ == "__main__":
    main()
