"""Check 5: strict=True load of reference K=1 checkpoints into new code."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from root.models.model import _build_model
from root.utils import dict_to_namespace

CKPTS = [
    ("/nas/manu/vjepa2/outputs/pred_in_dino_space_2ldiw9xk/last.pt", False),
    ("/nas/manu/vjepa2/outputs/patch_pred_dino_space_e6esmgmu/last.pt", True),
]

for path, use_patches in CKPTS:
    print(f"\n=== {path} (use_patch_tokens={use_patches}) ===")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = dict_to_namespace(ckpt["args"])
    # Disable cache paths that need args fields not required for model build
    args.cache_dino_feats = False
    args.load_cache_feats = True
    args.val_dataset_len = None
    # Force new field to default (matches old K=1 checkpoints)
    if not hasattr(args.encoder.rnn, "max_horizon"):
        args.encoder.rnn.max_horizon = 1
    if not hasattr(args.encoder.rnn, "horizon_weights"):
        args.encoder.rnn.horizon_weights = None

    model = _build_model(args, torch.device("cpu"))
    result = model.load_state_dict(ckpt["model"], strict=True)
    print(f"  load_state_dict(strict=True) result: {result}")
    print(f"  max_horizon={args.encoder.rnn.max_horizon}  use_patch_tokens={getattr(args, 'use_patch_tokens', False)}  OK")
