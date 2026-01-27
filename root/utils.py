from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

import torch

import random
import numpy as np
import wandb
import json
import argparse

def set_seed(seed_value):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def init_wandb(args, resume_run_id=None, fork=False):
    if args.wandb.logging and (not args.wandb.sweep) :
        if (resume_run_id is not None):
                if not fork:
                    wandb.init(id=resume_run_id, resume='must', entity=args.wandb.entity, project=args.wandb.project, config=OmegaConf.to_container(args, resolve=True))
                else:
                    wandb.init(
                        project=args.wandb.project,
                        entity=args.wandb.entity,
                        fork_from=f"{resume_run_id}?_step={str(args.wandb.restart_iter)}",
                        )
        else:
            wandb.init(entity=args.wandb.entity, project=args.wandb.project, config=vars(args))
        wandb.run.name = args.wandb.run_name

def int2mil(number):
    if abs(number) >= 100_000:
        formatted_number = "{:.1f}M".format(number / 1_000_000)
    else:
        formatted_number = str(number)
    return formatted_number


def trainable_params(model):
    return int2mil(sum(p.numel() for p in model.parameters() if p.requires_grad == True))


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    v = os.environ.get(name, None)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


@dataclass(frozen=True)
class TrainBatch:
    # `clips` is either:
    # - a list (length=num_clips) of tensors [B, C, T, H, W]
    # - a single tensor [B, C, T, H, W]
    clips: object
    labels: torch.Tensor
    ds_index: int

def _get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = _env_int("LOCAL_RANK", 0) or 0
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def _iter_batches(loader: Iterable) -> Iterable[TrainBatch]:
    # VideoDataset yields: (buffer, label, clip_indices)
    for clips, labels, _clip_idxs, index in loader:
        # labels may come in as list[str] or list[int] -> normalize here.
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        yield TrainBatch(clips=clips, labels=labels, ds_index=index)

def save_checkpoint(optimizer, args, epoch, global_step, state_dict, logger):
    ckpt_path = os.path.join(args.output_dir, f"best.pt")
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(namespace_to_dict(args), f, indent=4)
    to_save = {
        "epoch": epoch,
        "model": state_dict,
        "opt": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(to_save, ckpt_path)
    logger.info(f"Saved checkpoint: {ckpt_path}")

def namespace_to_dict(obj):
    if isinstance(obj, argparse.Namespace):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, dict):
        return {k: namespace_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [namespace_to_dict(v) for v in obj]
    else:
        return obj

def dict_to_namespace(data):
    if isinstance(data, dict):
        # Convert all values in the dict recursively, then wrap in Namespace
        return argparse.Namespace(**{k: dict_to_namespace(v) for k, v in data.items()})
    elif isinstance(data, list):
        # Handle lists by checking if they contain more dicts
        return [dict_to_namespace(v) for v in data]
    else:
        return data