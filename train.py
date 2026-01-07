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
import root.args as parser #import _parse_args, _resolve_sampling_kwargs
from root.model import _build_model
import root.utils as utils
import root.dataset as dataset
from root.ddp import _wrap_ddp, _ddp_mean, _is_distributed
from src.datasets.video_dataset import make_videodataset
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, get_logger
import wandb

logger = get_logger(__name__, force=True)

global_vars = {
    "best_acc": 0.0,
    "global_step": 0,
}

@torch.no_grad()
def run_validation(
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
    for batch in tqdm(utils._iter_batches(loader), total=len(loader), desc=f"Validating epoch {epoch}"):
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
        wandb.log({
            "eval/loss": mean_loss,
            "eval/acc": acc,
        }, step=global_vars["global_step"])
    model.train()
    return mean_loss, acc

def main(args) -> None:
    # torchrun compatibility: prefer env vars if available, but still call repo helper.
    env_rank = utils._env_int("RANK", None)
    env_world = utils._env_int("WORLD_SIZE", None)
    world_size, rank = init_distributed(port=args.dist_port, rank_and_world_size=(env_rank, env_world))

    device = utils._get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    is_master = rank == 0
    if is_master:
        os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Initialized device={device}, rank/world={rank}/{world_size}")

    # --- data
    transform = make_transforms(crop_size=args.crop_size)
    sampling_kwargs = parser._resolve_sampling_kwargs(args)


    train_ds, train_loader, train_sampler, val_ds, val_loader, val_sampler = dataset.get_loaders(args, transform, sampling_kwargs, rank, world_size, is_master)
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
    
    if args.init_eval and not args.debug:
        print("|RUNNING INITIAL VALIDATION...")
        _vloss, global_vars["best_acc"] = run_validation(model, val_loader, device, world_size, epoch=0, step=global_step, is_master=is_master)
        ###acc should be maintained. it will be used to save checkpoint if it is better than the previous one.

    for epoch in range(args.epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        it_time = AverageMeter()

        model.train()
        optimizer.zero_grad(set_to_none=True)

        for it, batch in tqdm(enumerate(utils._iter_batches(train_loader)), total=len(train_loader), desc=f"train epoch {epoch}"):
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

            # Logging ###???????
            loss_reduced = _ddp_mean(loss.detach() * max(args.grad_accum, 1)).item()
            loss_meter.update(loss_reduced, n=x.size(0))
            it_time.update((time.time() - t0) * 1000.0)

            if is_master:
                # logger.info(
                #     f"epoch={epoch} step={global_step} "
                #     f"loss={loss_meter.avg:.4f} iter_ms={it_time.avg:.1f} "
                #     f"bs={args.batch_size} world={world_size}"
                # )
                wandb.log({
                    "trainer/epoch": epoch,
                    "trainer/step": global_vars["global_step"],
                    "trainer/loss": loss_meter.avg,
                    "trainer/lr": optimizer.param_groups[0]["lr"],
                }, step=global_vars["global_step"])

            if (global_vars["global_step"]+1) % args.val_freq == 0:
                _vloss, acc = run_validation(model, val_loader, device, world_size, epoch=epoch, step=global_step, is_master=is_master)
                if acc > global_vars["best_acc"]:
                    global_vars["best_acc"] = acc
                    utils.save_checkpoint(optimizer, args, epoch, global_step, model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict(), logger)
                
            global_vars["global_step"] += 1

        if _is_distributed(world_size):
            dist.barrier()

    if _is_distributed(world_size):
        dist.destroy_process_group()



if __name__ == "__main__":
    utils.set_seed(42)
    args = parser.prepare_config()
    utils.init_wandb(args)
    main(args)

 
