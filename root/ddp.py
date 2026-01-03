from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


def _is_distributed(world_size: int) -> bool:
    return dist.is_available() and dist.is_initialized() and world_size > 1


def _ddp_mean(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        x = x.detach()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    return x


def _wrap_ddp(model: nn.Module, device: torch.device, world_size: int) -> nn.Module:
    if not _is_distributed(world_size):
        return model
    if device.type == "cuda":
        return DistributedDataParallel(model, device_ids=[device.index], static_graph=True)
    return DistributedDataParallel(model)


