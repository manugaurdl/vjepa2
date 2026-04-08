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
from eval_transfer import SimpleVideoDataset, iter_batches as ood_iter_batches
import wandb
import plotly.express as px
import plotly.graph_objects as go

logger = get_logger(__name__, force=True)

global_vars = {
    "best_acc": 0.0,           # used when action_classification=True
    "best_pred_loss": float("inf"),  # used when action_classification=False
    "global_step": 0,
}

from root.utils import compute_relative_state_shift


def _get_pred_error_l2(model: torch.nn.Module):
    pred_error_l2 = getattr(model, "pred_error_l2", None)
    if pred_error_l2 is None and isinstance(model, DistributedDataParallel):
        pred_error_l2 = getattr(model.module, "pred_error_l2", None)
    return pred_error_l2



@torch.no_grad()
def run_ood_eval(model, ood_loader, device, ssv2_pred_error_l2, is_master):
    """Collect per-timestep pred_error_l2 on OOD data and log comparison to wandb."""
    model.eval()
    ood_errors = []
    for x, y, ds_idx in tqdm(ood_iter_batches(ood_loader), total=len(ood_loader), desc="OOD eval"):
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        _ = model(x, ds_idx)
        pred_error_l2 = _get_pred_error_l2(model)
        if pred_error_l2 is not None and pred_error_l2.size(1) > 1:
            ood_errors.append(pred_error_l2[:, 1:].mean(dim=-1).mean(dim=0).cpu())  # (T-1,)
    if not ood_errors or ssv2_pred_error_l2 is None:
        return
    ood_mean = torch.stack(ood_errors).mean(0).tolist()
    if is_master and wandb.run:
        timesteps = list(range(1, len(ssv2_pred_error_l2) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timesteps, y=ssv2_pred_error_l2, mode="lines+markers", name="SSv2"))
        fig.add_trace(go.Scatter(x=timesteps, y=ood_mean, mode="lines+markers", name="OOD"))
        fig.update_layout(title="Pred Error L2: SSv2 vs OOD", xaxis_title="timestep", yaxis_title="pred_error_l2")
        ssv2_m = sum(ssv2_pred_error_l2) / len(ssv2_pred_error_l2)
        ood_m = sum(ood_mean) / len(ood_mean)
        wandb.log({
            "eval_ood/pred_error_l2": fig,
            "eval_ood/pred_error_ratio": ood_m / max(ssv2_m, 1e-8),
        }, step=int(global_vars["global_step"]))


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
    ce_loss_sum = torch.tensor(0.0, device=device)
    pred_loss_sum = torch.tensor(0.0, device=device)
    pred_loss_weight = getattr(args.encoder.rnn, "pred_loss_weight", 0.0)

    for batch in tqdm(utils._iter_batches(loader), total=len(loader), desc=f"Validating epoch {epoch}"):
        clips = batch.clips
        ds_index = batch.ds_index
        x = clips[0] if isinstance(clips, (list, tuple)) else clips
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = batch.labels.to(device=device, dtype=torch.long, non_blocking=True)
        logits = model(x, ds_index)
        if model.cache_dino_feats:
            continue
        if getattr(args, "action_classification", True):
            ce_loss_sum += F.cross_entropy(logits, y, reduction="sum")
        if getattr(args, "next_frame_pred", True):
            pred_error_l2 = _get_pred_error_l2(model)
            if pred_error_l2 is not None and pred_error_l2.size(1) > 1:
                pred_loss = pred_error_l2[:, 1:].mean(dim=(-1, -2))
                pred_loss_sum += pred_loss.sum()
        
        total += float(y.numel())
        if getattr(args, "action_classification", True):
            pred = logits.argmax(dim=1)
            correct += (pred == y).float().sum()
    
    if model.cache_dino_feats:
        save_dir = os.path.join(args.data_dir, "ssv2/dino_feats", args.dino_model.split("_")[-1])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "validation.pt")
        n_valid = len(loader.dataset)
        print(f"Saving dino feats to {save_path} (trimmed to {n_valid} valid rows)")
        torch.save(model.id_to_feat[:n_valid].clone(), save_path)
        exit()

    if _is_distributed(world_size):
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(ce_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(pred_loss_sum, op=dist.ReduceOp.SUM)

    acc = (correct / total.clamp_min(1.0)).item()
    mean_ce_loss = (ce_loss_sum / total.clamp_min(1.0)).item()
    mean_pred_loss = (pred_loss_sum / total.clamp_min(1.0)).item()
    if getattr(args, "action_classification", True):
        mean_total_loss = ((ce_loss_sum + pred_loss_sum * pred_loss_weight) / total.clamp_min(1.0)).item()
    else:
        mean_total_loss = (pred_loss_sum / total.clamp_min(1.0)).item()
    dynamics_logs = {}
    if hasattr(model, "collect_update_gates"):
        gate_means = model.update_gates.mean(0).tolist()
        update_norms = model.update_norms.mean(0).tolist()
        pred_error_l2s = getattr(model, "pred_error_l2s", None)
        pred_error_l2_mean = pred_error_l2s.mean(0).tolist()[1:] if pred_error_l2s is not None else None
        hidden_states = model.hidden_states
        r_novelty = model.r_novelty.mean(0).tolist()[1:] #skip first timestep
        memory_l2_shift, cos_sim, h_t_norm = compute_relative_state_shift(hidden_states)
        memory_l2_shift = memory_l2_shift.mean(0).tolist()
        cos_sim = cos_sim.mean(0).tolist()
        h_t_norm = h_t_norm.mean(0).tolist()

        def create_plotly_figure(data, title, y_label, x_label):
            data = [{x_label: t, y_label: g} for t, g in enumerate(data)]
            fig = px.line(data, x=x_label, y=y_label, title=title)
            return fig

        suffix = "{precision weighting}" if args.encoder.rnn.update_type == "surprise" else ""
        dynamics_logs = {
            "eval/r_novelty": create_plotly_figure(r_novelty, "Novelty Ratio over Time", "(u_novelty/u_total)", "timestep+1"),
            "eval/update_gate": create_plotly_figure(gate_means, f"Update Gate over Time {suffix}", "gate", "timestep"),
            "eval/update_norm": create_plotly_figure(update_norms, "Update Norm over Time", "update_norm", "timestep"),
            "eval/memory_l2": create_plotly_figure(memory_l2_shift, "Memory L2 Shift over Time", "l2_shift(h_t,h_{t-1})", "timestep+1"),
            "eval/cos_sim": create_plotly_figure(cos_sim, "Memory direction similarity over Time", "cos_sim(h_t,h_{t-1})", "timestep+1"),
            "eval/h_t_norm": create_plotly_figure(h_t_norm, "Memory norm over Time", "h_t_norm", "timestep+1"),
        }
        if pred_error_l2_mean is not None:
            dynamics_logs["eval/pred_error_l2"] = create_plotly_figure(
                pred_error_l2_mean, f"Prediction Error L2 over Time {suffix}", "pred_error_l2", "timestep"
            )

    if is_master and wandb.run:
        wandb.log({
            "eval/loss": mean_total_loss,
            "eval/total_loss": mean_total_loss,
            "eval/ce_loss": mean_ce_loss,
            "eval/pred_loss": mean_pred_loss,
            "eval/acc": acc,
            **dynamics_logs,
        }, step=int(global_vars["global_step"]))
    
    # Extract SSv2 pred_error_l2 for OOD comparison
    ssv2_pred_error_l2 = None
    if hasattr(model, "collect_update_gates"):
        pe = getattr(model, "pred_error_l2s", None)
        if pe is not None:
            ssv2_pred_error_l2 = pe.mean(0).tolist()[1:]
        model.collect_update_gates = False
    model.train()
    return mean_total_loss, acc, ssv2_pred_error_l2

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
    args.train_dataset_len = len(train_ds)
    args.val_dataset_len = len(val_ds)

    # --- OOD eval loader (optional)
    ood_loader = None
    ood_eval_csv = getattr(args, "ood_eval_csv", None)
    if ood_eval_csv is not None:
        ood_ds = SimpleVideoDataset(ood_eval_csv, frames_per_clip=args.eval_frames_per_clip, transform=eval_transform)
        ood_loader = torch.utils.data.DataLoader(
            ood_ds, batch_size=(args.val_batch_size or args.batch_size),
            shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False,
        )
        print(f"OOD eval: {len(ood_ds)} videos from {ood_eval_csv}")

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
        _vloss, _vacc, ssv2_pe = run_validation(model, val_loader, device, world_size, epoch=0, step=global_step, is_master=is_master)
        if getattr(args, "action_classification", True):
            global_vars["best_acc"] = _vacc
        else:
            global_vars["best_pred_loss"] = _vloss
        if ood_loader is not None:
            run_ood_eval(model, ood_loader, device, ssv2_pe, is_master)

    for epoch in range(args.epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        ce_loss_meter = AverageMeter()
        pred_loss_meter = AverageMeter()
        it_time = AverageMeter()

        model.train()
        optimizer.zero_grad(set_to_none=True)

        for it, batch in tqdm(enumerate(utils._iter_batches(train_loader)), total=len(train_loader), desc=f"train epoch {epoch}"):
            t0 = time.time()

            clips = batch.clips
            ds_index = batch.ds_index
            x = clips[0] if isinstance(clips, (list, tuple)) else clips
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = batch.labels.to(device=device, dtype=torch.long, non_blocking=True)
            logits = model(x, ds_index)
            if model.cache_dino_feats:
                continue
            pred_loss_weight = getattr(args.encoder.rnn, "pred_loss_weight", 0.0)
            ce_loss = torch.tensor(0.0, device=device)
            pred_loss = torch.tensor(0.0, device=device)
            if getattr(args, "action_classification", True):
                ce_loss = F.cross_entropy(logits, y)
            if getattr(args, "next_frame_pred", True):
                pred_error_l2 = _get_pred_error_l2(model)
                if pred_error_l2 is not None and pred_error_l2.size(1) > 1:
                    pred_loss = pred_error_l2[:, 1:].mean()
            if getattr(args, "action_classification", True):
                total_loss = ce_loss + pred_loss_weight * pred_loss
            else:
                total_loss = pred_loss
            total_loss = total_loss / max(args.grad_accum, 1)
            total_loss.backward()

            if (it + 1) % max(args.grad_accum, 1) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Logging ###???????
            ce_loss_reduced = _ddp_mean(ce_loss.detach()).item()
            if getattr(args, "action_classification", True):
                pred_loss_reduced = _ddp_mean((pred_loss * pred_loss_weight).detach()).item()
            else:
                pred_loss_reduced = _ddp_mean(pred_loss.detach()).item()
            total_loss_reduced = _ddp_mean(total_loss.detach() * max(args.grad_accum, 1)).item()
            loss_meter.update(total_loss_reduced, n=x.size(0))
            ce_loss_meter.update(ce_loss_reduced, n=x.size(0))
            pred_loss_meter.update(pred_loss_reduced, n=x.size(0))
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
                    "trainer/total_loss": loss_meter.avg,
                    "trainer/ce_loss": ce_loss_meter.avg,
                    "trainer/pred_loss": pred_loss_meter.avg,
                    "trainer/lr": optimizer.param_groups[0]["lr"],
                    "trainer/iter_ms_avg": it_time.avg,
                }, step=global_vars["global_step"])

            if (global_vars["global_step"]+1) % args.val_freq == 0:
                _vloss, acc, ssv2_pe = run_validation(model, val_loader, device, world_size, epoch=epoch, step=global_step, is_master=is_master)
                if ood_loader is not None:
                    run_ood_eval(model, ood_loader, device, ssv2_pe, is_master)
                model_state = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
                # best.pt save criterion: highest val acc when classifying, lowest val
                # pred_loss when training pred-only. Without this branch, pred-only runs
                # never saved a best.pt because acc stayed 0 and `acc > best_acc` was
                # always false (left 2ldiw9xk and e6esmgmu with only last.pt).
                if getattr(args, "action_classification", True):
                    is_best = acc > global_vars["best_acc"]
                    if is_best:
                        global_vars["best_acc"] = acc
                else:
                    is_best = _vloss < global_vars["best_pred_loss"]
                    if is_best:
                        global_vars["best_pred_loss"] = _vloss
                if is_best:
                    utils.save_checkpoint(optimizer, args, epoch, global_step, model_state, logger, ckpt_name="best")
                utils.save_checkpoint(optimizer, args, epoch, global_step, model_state, logger, ckpt_name="last")
                
            global_vars["global_step"] += 1

        if  model.cache_dino_feats:
            save_dir = os.path.join(args.data_dir, "ssv2/dino_feats", args.dino_model.split("_")[-1])
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "train.pt")
            n_train = len(train_loader.dataset)
            print(f"Saving dino feats to {save_path} (trimmed to {n_train} valid rows)")
            torch.save(model.id_to_feat[:n_train].clone(), save_path)
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

 
