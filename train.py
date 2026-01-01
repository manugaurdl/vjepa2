# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Minimal, hackable training entrypoint for video inputs.

This script intentionally stays small and composes functionality from the repo:
- Distributed init: `src.utils.distributed.init_distributed`
- Video dataset + loader: `src.datasets.video_dataset.make_videodataset`
- Video transforms: `app.vjepa.transforms.make_transforms`
- ViT backbone: `src.models.vision_transformer`

Example (single process):
  python train.py --data-path /path/to/train.csv --frames-per-clip 16 --frame-step 4 --num-classes 400

Example (DDP, 8 GPUs):
  torchrun --nproc_per_node 8 train.py --data-path /path/to/train.csv --frames-per-clip 16 --frame-step 4 --num-classes 400

Expected csv format (space-delimited, no header):
  /abs/or/rel/path/to/video.mp4 12
  /abs/or/rel/path/to/video_2.mp4 7
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

import src.models.vision_transformer as video_vit
from app.vjepa.transforms import make_transforms
from src.datasets.video_dataset import make_videodataset
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, get_logger


logger = get_logger(__name__, force=True)


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    v = os.environ.get(name, None)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _is_distributed(world_size: int) -> bool:
    return dist.is_available() and dist.is_initialized() and world_size > 1


def _ddp_mean(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        x = x.detach()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    return x


@dataclass(frozen=True)
class TrainBatch:
    # `clips` is either:
    # - a list (length=num_clips) of tensors [B, C, T, H, W]
    # - a single tensor [B, C, T, H, W]
    clips: object
    labels: torch.Tensor


class VideoViTClassifier(nn.Module):
    """
    Simple supervised head on top of the repo's `VisionTransformer` video backbone.
    """

    def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone returns token embeddings: [B, N, D]
        toks = self.backbone(x)
        pooled = toks.mean(dim=1)
        return self.head(pooled)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal video training script (hackable).")

    # --- I/O
    p.add_argument("--data-path", action="append", required=True, help="CSV or NPY path. Repeatable.")
    p.add_argument("--output-dir", default="./outputs/min_train", help="Where to write checkpoints/logs (rank 0).")

    # --- dataset / sampling
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--pin-mem", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--deterministic-loader", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--frames-per-clip", type=int, default=16)
    p.add_argument("--num-clips", type=int, default=1)
    p.add_argument("--random-clip-sampling", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-clip-overlap", action=argparse.BooleanOptionalAction, default=False)

    # Exactly one of (frame_step, fps, duration) must be specified by the dataset.
    p.add_argument(
        "--frame-step",
        type=int,
        default=None,
        help="Sample every k-th frame (mutually exclusive with --fps/--duration). Default: 4 if neither --fps nor --duration is set.",
    )
    p.add_argument("--fps", type=int, default=None, help="Target sampling fps (mutually exclusive with --frame-step/--duration).")
    p.add_argument("--duration", type=float, default=None, help="Clip duration in seconds (mutually exclusive with --frame-step/--fps).")

    p.add_argument("--crop-size", type=int, default=224, help="Spatial crop size used by repo transforms.")

    # --- model
    p.add_argument("--model-name", type=str, default="vit_small", help="One of src.models.vision_transformer.*")
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--tubelet-size", type=int, default=2)
    p.add_argument("--use-sdpa", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-rope", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--num-classes", type=int, required=True, help="Number of classes (labels in CSV must be integer in [0, num_classes)).")

    # --- optimization
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-accum", type=int, default=1)

    # --- logging / checkpointing
    p.add_argument("--log-freq", type=int, default=20)
    p.add_argument("--save-every-epochs", type=int, default=1)

    # --- distributed
    p.add_argument("--dist-port", type=int, default=37129)

    return p.parse_args()


def _resolve_sampling_kwargs(args: argparse.Namespace) -> dict:
    # The dataset enforces exactly one of (fps, duration, frame_step) is not None.
    if args.fps is None and args.duration is None and args.frame_step is None:
        args.frame_step = 4

    specified = [args.frame_step is not None, args.fps is not None, args.duration is not None]
    if sum(specified) != 1:
        raise ValueError("Must specify exactly one of: --frame-step, --fps, --duration (set the others to None).")
    return dict(frame_step=args.frame_step, fps=args.fps, duration=args.duration)


def _get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = _env_int("LOCAL_RANK", 0) or 0
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def _build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.model_name not in video_vit.__dict__:
        raise ValueError(f"Unknown --model-name={args.model_name!r}. Must exist in src.models.vision_transformer.")

    backbone = video_vit.__dict__[args.model_name](
        img_size=(args.crop_size, args.crop_size),
        patch_size=args.patch_size,
        num_frames=args.frames_per_clip,
        tubelet_size=args.tubelet_size,
        use_sdpa=args.use_sdpa,
        use_rope=args.use_rope,
    )
    embed_dim = getattr(backbone, "embed_dim", None)
    if embed_dim is None:
        raise RuntimeError("Backbone did not expose embed_dim; expected VisionTransformer-like module.")

    model = VideoViTClassifier(backbone=backbone, embed_dim=int(embed_dim), num_classes=args.num_classes)
    return model.to(device)


def _wrap_ddp(model: nn.Module, device: torch.device, world_size: int) -> nn.Module:
    if not _is_distributed(world_size):
        return model
    if device.type == "cuda":
        return DistributedDataParallel(model, device_ids=[device.index], static_graph=True)
    return DistributedDataParallel(model)


def _iter_batches(loader: Iterable) -> Iterable[TrainBatch]:
    # VideoDataset yields: (buffer, label, clip_indices)
    for clips, labels, _clip_idxs in loader:
        # labels may come in as list[str] or list[int] -> normalize here.
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        yield TrainBatch(clips=clips, labels=labels)


def main() -> None:
    args = _parse_args()

    # torchrun compatibility: prefer env vars if available, but still call repo helper.
    env_rank = _env_int("RANK", None)
    env_world = _env_int("WORLD_SIZE", None)
    world_size, rank = init_distributed(port=args.dist_port, rank_and_world_size=(env_rank, env_world))

    device = _get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    is_master = rank == 0
    if is_master:
        os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Initialized device={device}, rank/world={rank}/{world_size}")

    # --- data
    transform = make_transforms(crop_size=args.crop_size)
    sampling_kwargs = _resolve_sampling_kwargs(args)

    _dataset, loader, sampler = make_videodataset(
        data_paths=args.data_path,
        batch_size=args.batch_size,
        frames_per_clip=args.frames_per_clip,
        num_clips=args.num_clips,
        random_clip_sampling=args.random_clip_sampling,
        allow_clip_overlap=args.allow_clip_overlap,
        transform=transform,
        shared_transform=None,
        rank=rank,
        world_size=world_size,
        collator=None,
        drop_last=args.drop_last,
        num_workers=args.num_workers,
        pin_mem=args.pin_mem,
        persistent_workers=args.persistent_workers,
        deterministic=args.deterministic_loader,
        log_dir=(os.path.join(args.output_dir, "dataloader_logs") if is_master else None),
        **sampling_kwargs,
    )

    # --- model / opt
    model = _build_model(args, device)
    model = _wrap_ddp(model, device, world_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- train
    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        it_time = AverageMeter()

        model.train()
        optimizer.zero_grad(set_to_none=True)

        for it, batch in enumerate(_iter_batches(loader)):
            t0 = time.time()

            clips = batch.clips
            x = clips[0] if isinstance(clips, (list, tuple)) else clips
            x = x.to(device, non_blocking=True)
            y = batch.labels.to(device=device, dtype=torch.long, non_blocking=True)

            logits = model(x)
            loss = F.cross_entropy(logits, y) / max(args.grad_accum, 1)
            loss.backward()

            if (it + 1) % max(args.grad_accum, 1) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Logging
            loss_reduced = _ddp_mean(loss.detach() * max(args.grad_accum, 1)).item()
            loss_meter.update(loss_reduced, n=x.size(0))
            it_time.update((time.time() - t0) * 1000.0)

            if is_master and (global_step % args.log_freq == 0):
                logger.info(
                    f"epoch={epoch} step={global_step} "
                    f"loss={loss_meter.avg:.4f} iter_ms={it_time.avg:.1f} "
                    f"bs={args.batch_size} world={world_size}"
                )

            global_step += 1

        if is_master and args.save_every_epochs > 0 and ((epoch + 1) % args.save_every_epochs == 0):
            ckpt_path = os.path.join(args.output_dir, f"ckpt_epoch_{epoch:04d}.pt")
            to_save = {
                "epoch": epoch,
                "model": model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict(),
                "opt": optimizer.state_dict(),
                "args": vars(args),
            }
            torch.save(to_save, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

        if _is_distributed(world_size):
            dist.barrier()

    if _is_distributed(world_size):
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

 
