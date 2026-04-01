"""Precompute DINOv2 patch-level features for SSv2."""

import argparse
import os
import torch
from tqdm import tqdm
from app.vjepa.transforms import make_transforms
from src.datasets.video_dataset import make_videodataset


def extract_patch_features(dino, frames):
    """(B*T, C, H, W) -> (B*T, S, D) patch tokens."""
    with torch.no_grad():
        feats = dino.forward_features(frames)
        return feats["x_norm_patchtokens"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--dino_model", type=str, default="dinov2_vits14")
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--frames_per_clip", type=int, default=8)
    p.add_argument("--frame_step", type=int, default=4)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    dino = torch.hub.load("facebookresearch/dinov2", args.dino_model, pretrained=True)
    dino = dino.to(device).eval()

    # Get patch count and dim from dummy forward
    with torch.no_grad():
        dummy = torch.randn(1, 3, args.crop_size, args.crop_size, device=device)
        dummy_out = extract_patch_features(dino, dummy)
        n_patches, dino_dim = dummy_out.shape[1], dummy_out.shape[2]
    print(f"Patches: {n_patches}, dim: {dino_dim}")

    transform = make_transforms(mode="eval", crop_size=args.crop_size)

    splits = [
        ("train", os.path.join(args.data_dir, "ssv2/data/train.csv"), 168913),
        ("validation", os.path.join(args.data_dir, "ssv2/data/validation.csv"), 24777),
    ]

    # Minimal args namespace for make_videodataset
    ds_args = argparse.Namespace(
        data_dir=args.data_dir,
        dino_model=args.dino_model,
        debug=False,
    )

    for split_name, csv_path, n_videos in splits:
        ds, loader, _ = make_videodataset(
            data_paths=csv_path,
            args=ds_args,
            batch_size=args.batch_size,
            frames_per_clip=args.frames_per_clip,
            num_clips=1,
            random_clip_sampling=False,
            allow_clip_overlap=False,
            transform=transform,
            shared_transform=None,
            rank=0,
            world_size=1,
            collator=None,
            drop_last=False,
            num_workers=args.num_workers,
            pin_mem=True,
            persistent_workers=False,
            deterministic=True,
            log_dir=None,
            debug=False,
            uniform_sampling=True,
            shuffle=False,
            load_cache_feats=False,
            frame_step=args.frame_step,
        )

        all_feats = torch.zeros(n_videos, args.frames_per_clip, n_patches, dino_dim, dtype=torch.float16)
        idx = 0

        for batch in tqdm(loader, desc=f"{split_name}"):
            clips, labels, clip_idxs, index = batch
            x = clips[0] if isinstance(clips, (list, tuple)) else clips
            B, C, T, H, W = x.shape
            frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).to(device)
            patch_feats = extract_patch_features(dino, frames)  # (B*T, S, D)
            patch_feats = patch_feats.reshape(B, T, n_patches, dino_dim)
            all_feats[idx:idx + B] = patch_feats.cpu().half()
            idx += B

        save_dir = os.path.join(args.data_dir, "ssv2/dino_feats", args.dino_model.split("_")[-1])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{split_name}_patches.pt")
        print(f"Saving {save_path} — shape {all_feats.shape}")
        torch.save(all_feats, save_path)


if __name__ == "__main__":
    main()
