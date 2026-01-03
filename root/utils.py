from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

import torch


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


def _get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = _env_int("LOCAL_RANK", 0) or 0
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def _iter_batches(loader: Iterable) -> Iterable[TrainBatch]:
    # VideoDataset yields: (buffer, label, clip_indices)
    for clips, labels, _clip_idxs in loader:
        # labels may come in as list[str] or list[int] -> normalize here.
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        yield TrainBatch(clips=clips, labels=labels)


