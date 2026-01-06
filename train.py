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

import os
import time
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from app.vjepa.transforms import make_transforms
from root.argparse import _parse_args, _resolve_sampling_kwargs
from root.model import _build_model
from root.utils import _env_int, _get_device, _iter_batches
from root.ddp import _wrap_ddp, _ddp_mean, _is_distributed
from src.datasets.video_dataset import make_videodataset
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, get_logger


logger = get_logger(__name__, force=True)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    world_size: int,
    *,
    epoch: int,
    step: int,
    is_master: bool,
) -> tuple[float, float]:
    model.eval()
    correct = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)
    loss_sum = torch.tensor(0.0, device=device)
    for batch in tqdm(_iter_batches(loader), total=len(loader), desc=f"Validating epoch {epoch}"):
        clips = batch.clips
        x = clips[0] if isinstance(clips, (list, tuple)) else clips
        x = x.to(device, non_blocking=True)
        y = batch.labels.to(device=device, dtype=torch.long, non_blocking=True)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum")
        pred = logits.argmax(dim=1)
        correct += (pred == y).float().sum()
        total += float(y.numel())

    if _is_distributed(world_size):
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)

    acc = (correct / total.clamp_min(1.0)).item()
    mean_loss = (loss_sum / total.clamp_min(1.0)).item()
    if is_master:
        print(f"val epoch_num={epoch} step_num={step} mean_loss={mean_loss:.4f} mean_acc={acc:.4f}", flush=True)
    model.train()
    return mean_loss, acc

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

    _vd, val_loader, val_sampler = make_videodataset(
        data_paths=args.val_data_path,
        batch_size=(args.val_batch_size or args.batch_size),
        frames_per_clip=args.frames_per_clip,
        num_clips=args.num_clips,
        random_clip_sampling=False,
        allow_clip_overlap=args.allow_clip_overlap,
        transform=transform,
        shared_transform=None,
        rank=rank,
        world_size=world_size,
        collator=None,
        drop_last=False,
        num_workers=args.num_workers,
        pin_mem=args.pin_mem,
        persistent_workers=args.persistent_workers,
        deterministic=True,
        log_dir=None,
        **sampling_kwargs,
    )

    # --- model / opt
    model = _build_model(args, device)
    model = _wrap_ddp(model, device, world_size)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --- train
    global_step = 0
    
    if not args.debug:
        _vloss, best_acc = validate(model, val_loader, device, world_size, epoch=0, step=global_step, is_master=is_master)
        ###acc should be maintained. it will be used to save checkpoint if it is better than the previous one.

    
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

 
