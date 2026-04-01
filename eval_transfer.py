"""
Standalone evaluation script for transfer linear probe + OOD prediction decay.

Usage:
  # Transfer probe (RNN state → UCF101)
  python eval_transfer.py --checkpoint best.pt --mode probe \
      --train_csv /nas/manu/ucf101/data/train.csv \
      --data_csv /nas/manu/ucf101/data/test.csv \
      --num_classes 101 --output_dir outputs/eval_probe

  # Transfer probe (DINO mean-pool baseline)
  python eval_transfer.py --checkpoint best.pt --mode probe --baseline \
      --train_csv /nas/manu/ucf101/data/train.csv \
      --data_csv /nas/manu/ucf101/data/test.csv \
      --num_classes 101 --output_dir outputs/eval_probe_baseline

  # OOD prediction decay curve
  python eval_transfer.py --checkpoint best.pt --mode decay \
      --data_csv /nas/manu/ucf101/data/test.csv \
      --ssv2_val_csv /nas/manu/ssv2/data/validation.csv \
      --output_dir outputs/eval_decay

  # Both
  python eval_transfer.py --checkpoint best.pt --mode both \
      --train_csv /nas/manu/ucf101/data/train.csv \
      --data_csv /nas/manu/ucf101/data/test.csv \
      --ssv2_val_csv /nas/manu/ssv2/data/validation.csv \
      --num_classes 101 --output_dir outputs/eval
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import warnings

ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from app.vjepa.transforms import make_transforms
from root.models.model import DinoFrameEncoder, _build_model
from root.utils import dict_to_namespace, set_seed
from train import compute_relative_state_shift

# ---------------------------------------------------------------------------
# SimpleVideoDataset — bypasses the SSv2-hardcoded VideoDataset
# ---------------------------------------------------------------------------

class SimpleVideoDataset(Dataset):
    """Reads a space-delimited CSV with absolute video paths and integer labels.

    Returns (clip_list, label, clip_indices, index) matching the interface
    expected by root.utils._iter_batches.
    """

    def __init__(self, csv_path, frames_per_clip=8, transform=None):
        from decord import VideoReader, cpu  # noqa: F811

        self.frames_per_clip = frames_per_clip
        self.transform = transform
        self._VideoReader = VideoReader
        self._cpu = cpu

        self.samples = []
        self.labels = []
        with open(csv_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                self.samples.append(parts[0])
                self.labels.append(int(parts[1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        label = self.labels[index]

        try:
            vr = self._VideoReader(path, num_threads=-1, ctx=self._cpu(0))
        except Exception:
            warnings.warn(f"Failed to load {path}, returning random sample")
            return self[np.random.randint(len(self))]

        n_frames = len(vr)
        if n_frames == 0:
            return self[np.random.randint(len(self))]

        # Uniform sampling (matches VideoDataset.loadvideo_decord uniform path)
        indices = np.linspace(0, n_frames - 1, num=self.frames_per_clip)
        indices = np.clip(indices, 0, n_frames - 1).astype(np.int64)

        buffer = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)

        if self.transform is not None:
            buffer = self.transform(buffer)  # eval: (C, T, H, W)

        clip_indices = [indices]
        return [buffer], label, clip_indices, index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(ckpt_path, device, override_num_classes=None):
    """Load a trained DinoFrameEncoder from a checkpoint.

    Returns (model, saved_args).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved_args = dict_to_namespace(ckpt["args"])

    # Override fields that don't apply at eval time
    saved_args.val_dataset_len = None
    saved_args.load_cache_feats = False
    saved_args.cache_dino_feats = False
    if not hasattr(saved_args, "use_patch_tokens"):
        saved_args.use_patch_tokens = False

    if override_num_classes is not None:
        saved_args.num_classes = override_num_classes

    model = _build_model(saved_args, device)
    model.load_state_dict(ckpt["model"], strict=False)
    return model, saved_args


def build_baseline_model(device):
    """Build a DINO mean-pool baseline (no RNN, just frozen DINO + linear head)."""
    baseline_args = dict_to_namespace({
        "encoder": {"type": "linear"},
        "pooling": "mean",
        "num_classes": 1,  # placeholder, head gets replaced anyway
        "dino_repo": "facebookresearch/dinov2",
        "dino_model": "dinov2_vits14",
        "dino_pretrained": True,
        "freeze_dino": True,
        "frames_per_clip": 8,
        "eval_frames_per_clip": 8,
        "cache_dino_feats": False,
        "load_cache_feats": False,
        "val_dataset_len": None,
        "action_classification": True,
        "use_patch_tokens": False,
    })
    model = _build_model(baseline_args, device)
    return model


def make_loader(csv_path, frames_per_clip, crop_size, batch_size, num_workers, mode="eval"):
    transform = make_transforms(mode=mode, crop_size=crop_size)
    ds = SimpleVideoDataset(csv_path, frames_per_clip=frames_per_clip, transform=transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    return ds, loader


def iter_batches(loader):
    """Yield (x, y, ds_index) from a DataLoader over SimpleVideoDataset."""
    for clips, labels, _clip_idxs, index in loader:
        x = clips[0] if isinstance(clips, (list, tuple)) else clips
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        yield x, labels, index


# ---------------------------------------------------------------------------
# Eval 1: Transfer Linear Probe
# ---------------------------------------------------------------------------

def run_probe(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    if args.baseline:
        print("=== DINO mean-pool baseline ===")
        model = build_baseline_model(device)
    else:
        print("=== RNN transfer probe ===")
        model, _ = load_model_from_checkpoint(args.checkpoint, device)

    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Replace classification head
    if model.encoder_type == "rnn":
        head_dim = model.head.in_features if model.head is not None else 384
    else:
        head_dim = 384  # DINO vits14 dim
    model.head = nn.Linear(head_dim, args.num_classes).to(device)
    # Unfreeze only the new head
    for p in model.head.parameters():
        p.requires_grad_(True)
    model.action_classification = True

    # Data
    train_ds, train_loader = make_loader(
        args.train_csv, args.frames_per_clip, args.crop_size,
        args.batch_size, args.num_workers, mode="train",
    )
    val_ds, val_loader = make_loader(
        args.data_csv, args.frames_per_clip, args.crop_size,
        args.batch_size, args.num_workers, mode="eval",
    )
    print(f"Train: {len(train_ds)} videos, Val: {len(val_ds)} videos")

    # Optimizer (only head params)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=0.01)

    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        # Keep DINO and encoder frozen
        if model.dino is not None:
            model.dino.eval()

        loss_sum, n_samples = 0.0, 0
        for x, y, ds_idx in tqdm(iter_batches(train_loader), total=len(train_loader), desc=f"Probe train epoch {epoch}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, dtype=torch.long, non_blocking=True)

            logits = model(x, ds_idx)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * x.size(0)
            n_samples += x.size(0)

        train_loss = loss_sum / max(n_samples, 1)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y, ds_idx in tqdm(iter_batches(val_loader), total=len(val_loader), desc=f"Probe val epoch {epoch}"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, dtype=torch.long, non_blocking=True)
                logits = model(x, ds_idx)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

        acc = correct / max(total, 1)
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_acc={acc:.4f}  best={best_acc:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "mode": "baseline" if args.baseline else "rnn",
        "dataset_csv": args.data_csv,
        "num_classes": args.num_classes,
        "epochs": args.epochs,
        "lr": args.lr,
        "best_acc": best_acc,
        "final_acc": acc,
    }
    result_path = os.path.join(args.output_dir, "probe_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")
    return result


# ---------------------------------------------------------------------------
# Eval 2: OOD Prediction Error Decay Curve
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_decay_stats(model, loader, device, dataset_len, frames_per_clip):
    """Run inference and collect per-timestep prediction error + state stats."""
    model.eval()

    pred_error_l2s = torch.zeros(dataset_len, frames_per_clip)
    update_norms = torch.zeros(dataset_len, frames_per_clip)
    r_novelty = torch.zeros(dataset_len, frames_per_clip)
    hidden_states = torch.zeros(dataset_len, frames_per_clip, 384)

    # Temporarily set up collection tensors on the model
    model_ref = model.module if hasattr(model, "module") else model
    model_ref.update_gates = torch.zeros(dataset_len, frames_per_clip)
    model_ref.update_norms = torch.zeros(dataset_len, frames_per_clip)
    model_ref.r_novelty = torch.zeros(dataset_len, frames_per_clip)
    model_ref.hidden_states = torch.zeros(dataset_len, frames_per_clip, 384)
    model_ref.pred_error_l2s = torch.zeros(dataset_len, frames_per_clip)
    model_ref.collect_update_gates = True

    for x, y, ds_idx in tqdm(iter_batches(loader), total=len(loader), desc="Collecting decay stats"):
        x = x.to(device, non_blocking=True)
        _ = model(x, ds_idx)

    # Gather results
    pred_error_l2s = model_ref.pred_error_l2s.clone()
    update_norms = model_ref.update_norms.clone()
    r_novelty = model_ref.r_novelty.clone()
    hidden_states = model_ref.hidden_states.clone()

    model_ref.collect_update_gates = False

    return {
        "pred_error_l2": pred_error_l2s,
        "update_norms": update_norms,
        "r_novelty": r_novelty,
        "hidden_states": hidden_states,
    }


def plot_comparison(ssv2_stats, ood_stats, output_dir, ood_name="UCF101"):
    """Generate comparison plots for SSv2 vs OOD."""
    os.makedirs(output_dir, exist_ok=True)

    plots = [
        ("pred_error_l2", "Prediction Error L2", "pred_error_l2"),
        ("update_norms", "Update Norm", "update_norm"),
        ("r_novelty", "Novelty Ratio", "r_novelty"),
    ]

    for key, ylabel, filename in plots:
        ssv2_mean = ssv2_stats[key][:, 1:].mean(0).numpy()  # skip t=0
        ood_mean = ood_stats[key][:, 1:].mean(0).numpy()

        fig, ax = plt.subplots(figsize=(8, 5))
        timesteps = np.arange(1, len(ssv2_mean) + 1)
        ax.plot(timesteps, ssv2_mean, "o-", label="SSv2 (in-distribution)", linewidth=2)
        ax.plot(timesteps, ood_mean, "s--", label=f"{ood_name} (OOD)", linewidth=2)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}: SSv2 vs {ood_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{filename}_decay.png"), dpi=150)
        plt.close(fig)
        print(f"Saved {filename}_decay.png")

    # Hidden state norm plot
    for name, stats in [("SSv2", ssv2_stats), (ood_name, ood_stats)]:
        h = stats["hidden_states"]
        h_norm = torch.norm(h, p=2, dim=-1)  # (N, T)
        stats["h_norm"] = h_norm

    fig, ax = plt.subplots(figsize=(8, 5))
    ssv2_hn = ssv2_stats["h_norm"].mean(0).numpy()
    ood_hn = ood_stats["h_norm"].mean(0).numpy()
    timesteps = np.arange(len(ssv2_hn))
    ax.plot(timesteps, ssv2_hn, "o-", label="SSv2", linewidth=2)
    ax.plot(timesteps, ood_hn, "s--", label=ood_name, linewidth=2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Hidden State L2 Norm")
    ax.set_title(f"Hidden State Norm: SSv2 vs {ood_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "hidden_state_norm_decay.png"), dpi=150)
    plt.close(fig)
    print("Saved hidden_state_norm_decay.png")


def run_decay(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (keep original SSv2 head — we only care about pred_error_l2)
    model, saved_args = load_model_from_checkpoint(args.checkpoint, device)
    model.eval()

    # OOD dataset
    ood_ds, ood_loader = make_loader(
        args.data_csv, args.frames_per_clip, args.crop_size,
        args.batch_size, args.num_workers,
    )

    # SSv2 val dataset
    ssv2_ds, ssv2_loader = make_loader(
        args.ssv2_val_csv, args.frames_per_clip, args.crop_size,
        args.batch_size, args.num_workers,
    )

    print(f"OOD dataset: {len(ood_ds)} videos")
    print(f"SSv2 val: {len(ssv2_ds)} videos")

    # Collect stats
    print("\n--- SSv2 (in-distribution) ---")
    ssv2_stats = collect_decay_stats(model, ssv2_loader, device, len(ssv2_ds), args.frames_per_clip)

    print("\n--- OOD ---")
    ood_stats = collect_decay_stats(model, ood_loader, device, len(ood_ds), args.frames_per_clip)

    # Save raw tensors
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(ssv2_stats, os.path.join(args.output_dir, "decay_ssv2.pt"))
    torch.save(ood_stats, os.path.join(args.output_dir, "decay_ood.pt"))
    print(f"Raw tensors saved to {args.output_dir}")

    # Plot
    plot_comparison(ssv2_stats, ood_stats, args.output_dir)

    # Print summary
    ssv2_pe = ssv2_stats["pred_error_l2"][:, 1:].mean().item()
    ood_pe = ood_stats["pred_error_l2"][:, 1:].mean().item()
    print(f"\nMean pred_error_l2 — SSv2: {ssv2_pe:.4f}, OOD: {ood_pe:.4f}, ratio: {ood_pe / max(ssv2_pe, 1e-8):.2f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Transfer probe + OOD decay eval")

    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pt checkpoint")
    p.add_argument("--mode", type=str, required=True, choices=["probe", "decay", "both"])

    # Data
    p.add_argument("--data_csv", type=str, required=True, help="OOD eval/test CSV (absolute paths)")
    p.add_argument("--train_csv", type=str, default=None, help="OOD train CSV (for probe mode)")
    p.add_argument("--ssv2_val_csv", type=str, default=None, help="SSv2 validation CSV (for decay mode)")
    p.add_argument("--num_classes", type=int, default=101, help="Number of classes in OOD dataset")

    # Model / data
    p.add_argument("--frames_per_clip", type=int, default=8)
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--baseline", action="store_true", help="Use DINO mean-pool baseline instead of RNN")

    # Training (probe mode)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)

    # Output
    p.add_argument("--output_dir", type=str, default="outputs/eval_transfer")

    return p.parse_args()


if __name__ == "__main__":
    set_seed(42)
    args = parse_args()

    if args.mode in ("probe", "both"):
        if args.train_csv is None:
            raise ValueError("--train_csv required for probe mode")
        run_probe(args)

    if args.mode in ("decay", "both"):
        if args.ssv2_val_csv is None:
            raise ValueError("--ssv2_val_csv required for decay mode")
        run_decay(args)
