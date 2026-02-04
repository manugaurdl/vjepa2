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
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import time
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from app.vjepa.transforms import make_transforms
import root.args as parser #import _parse_args, _resolve_sampling_kwargs
from root.models.model import _build_model
import root.utils as utils
import root.dataset as dataset
from root.ddp import _wrap_ddp, _ddp_mean, _is_distributed
from src.datasets.video_dataset import make_videodataset
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, get_logger
import wandb
import plotly.express as px

logger = get_logger(__name__, force=True)

global_vars = {
    "best_acc": 0.0,
    "global_step": 0,
}

def compute_relative_state_shift(hidden_states, epsilon=1e-8):
    """
    arg: hidden_states: (N, T, D)
    returns: relative_shift: (N, T-1, 1); l2(H_t - H_{t-1})/l2(H_{t-1})
    """
    h_t = hidden_states[:, 1:, :]
    h_prev = hidden_states[:, :-1, :]
    cos_sim = F.cosine_similarity(h_t, h_prev, dim=-1, eps=epsilon)
    delta = h_t - h_prev
    delta_norm = torch.norm(delta, p=2, dim=-1)
    prev_norm = torch.norm(h_prev, p=2, dim=-1)
    h_t_norm = torch.norm(h_t, p=2, dim=-1)
    relative_shift = delta_norm / (prev_norm + epsilon)
    return relative_shift, cos_sim, h_t_norm



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
    if hasattr(model, "collect_update_gates"):
        model.collect_update_gates = True
    correct = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)
    loss_sum = torch.tensor(0.0, device=device)

    for batch in tqdm(utils._iter_batches(loader), total=len(loader), desc=f"Validating epoch {epoch}"):
        clips = batch.clips
        ds_index = batch.ds_index
        x = clips[0] if isinstance(clips, (list, tuple)) else clips
        x = x.to(device, non_blocking=True)
        y = batch.labels.to(device=device, dtype=torch.long, non_blocking=True)
        logits = model(x, ds_index)
        if model.cache_dino_feats:
            continue
        loss_sum += F.cross_entropy(logits, y, reduction="sum")
        pred = logits.argmax(dim=1)
        correct += (pred == y).float().sum()
        total += float(y.numel())
    
    if model.cache_dino_feats:
        save_dir = os.path.join(args.data_dir, "ssv2/dino_feats", args.dino_model.split("_")[-1])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "validation.pt")
        print(f"Saving dino feats to {save_path}")
        torch.save(model.id_to_feat, save_path)                                              
        exit()

    if _is_distributed(world_size):
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)

    acc = (correct / total.clamp_min(1.0)).item()
    mean_loss = (loss_sum / total.clamp_min(1.0)).item()
    gate_means = model.update_gates.mean(0).tolist()
    update_norms = model.update_norms.mean(0).tolist()
    hidden_states = model.hidden_states
    r_novelty = model.r_novelty.mean(0).tolist()[1:] #skip first timestep
    memory_l2_shift, cos_sim, h_t_norm = compute_relative_state_shift(hidden_states)
    memory_l2_shift = memory_l2_shift.mean(0).tolist() #skip first shift (not computatble)
    cos_sim = cos_sim.mean(0).tolist() #skip first shift (not computatble)
    h_t_norm = h_t_norm.mean(0).tolist()  # skip t=0, not useful
    if is_master and wandb.run:
        def create_plotly_figure(data, title, y_label, x_label):
            data = [{x_label: t, y_label: g} for t, g in enumerate(data)] # dataframe like list
            fig = px.line(data, x=x_label, y=y_label, title=title) #plotly line plot
            return fig
        
        suffix = "precision weighting" if args.encoder.rnn.update_type == "surprise" else ""
        fig_update_gate = create_plotly_figure(gate_means, f"Update Gate over Time {suffix}", "gate", "timestep")
        fig_update_norm = create_plotly_figure(update_norms, "Update Norm over Time", "update_norm", "timestep")
        fig_r_novelty = create_plotly_figure(r_novelty, "Novelty Ratio over Time", "(u_novelty/u_total)", "timestep+1")
        fig_memory_l2 = create_plotly_figure(memory_l2_shift, "Memory L2 Shift over Time", "l2_shift(h_t,h_{t-1})", "timestep+1")
        fig_cos_sim = create_plotly_figure(cos_sim, "Memory direction similarity over Time", "cos_sim(h_t,h_{t-1})", "timestep+1")
        fig_h_t_norm = create_plotly_figure(h_t_norm, "Memory norm over Time", "h_t_norm", "timestep+1")

        wandb.log({
            "eval/loss": mean_loss,
            "eval/acc": acc,
            "eval/r_novelty": fig_r_novelty,
            "eval/update_gate": fig_update_gate,
            "eval/update_norm": fig_update_norm,
            "eval/memory_l2": fig_memory_l2,
            "eval/cos_sim": fig_cos_sim,
            "eval/h_t_norm": fig_h_t_norm,
        }, step=int(global_vars["global_step"]))
    
    if hasattr(model, "collect_update_gates"):
        model.collect_update_gates = False
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
    if is_master and wandb.run:
        args.output_dir = os.path.join(args.output_dir, args.wandb.run_name + "_" + wandb.run.id)
        os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Initialized device={device}, rank/world={rank}/{world_size}")

    # --- data
    if args.cache_dino_feats:
        mode = "eval" #when load_cache_feats=true, transforms not called, automatically eval mode
    else:
        mode = "train"
    train_transform = make_transforms(mode=mode, crop_size=args.crop_size)
    eval_transform = make_transforms(mode="eval", crop_size=args.crop_size)
    sampling_kwargs = parser._resolve_sampling_kwargs(args)


    train_ds, train_loader, train_sampler, val_ds, val_loader, val_sampler = dataset.get_loaders(args, train_transform, eval_transform, sampling_kwargs, rank, world_size, is_master)
    args.val_dataset_len = len(val_ds)

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
    
    if args.init_eval:
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
            ds_index = batch.ds_index
            x = clips[0] if isinstance(clips, (list, tuple)) else clips
            x = x.to(device, non_blocking=True)
            y = batch.labels.to(device=device, dtype=torch.long, non_blocking=True)
            logits = model(x, ds_index)
            if model.cache_dino_feats:
                continue
            loss = F.cross_entropy(logits, y) / max(args.grad_accum, 1)
            loss.backward()

            if (it + 1) % max(args.grad_accum, 1) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Logging ###???????
            loss_reduced = _ddp_mean(loss.detach() * max(args.grad_accum, 1)).item()
            loss_meter.update(loss_reduced, n=x.size(0))
            it_time.update((time.time() - t0) * 1000.0)

            if is_master and args.wandb.logging:
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
                    "trainer/iter_ms_avg": it_time.avg,
                }, step=global_vars["global_step"])

            if (global_vars["global_step"]+1) % args.val_freq == 0:
                _vloss, acc = run_validation(model, val_loader, device, world_size, epoch=epoch, step=global_step, is_master=is_master)
                if acc > global_vars["best_acc"]:
                    global_vars["best_acc"] = acc
                    utils.save_checkpoint(optimizer, args, epoch, global_step, model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict(), logger)
                
            global_vars["global_step"] += 1

        if  model.cache_dino_feats:
            save_dir = os.path.join(args.data_dir, "ssv2/dino_feats", args.dino_model.split("_")[-1])
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "train.pt")
            print(f"Saving dino feats to {save_path}")
            torch.save(model.id_to_feat, save_path)                                              
            exit()
        if _is_distributed(world_size):
            dist.barrier()

    if _is_distributed(world_size):
        dist.destroy_process_group()



if __name__ == "__main__":
    utils.set_seed(42)
    args = parser.prepare_config()
    if args.wandb.logging:
        utils.init_wandb(args)
    main(args)

 
