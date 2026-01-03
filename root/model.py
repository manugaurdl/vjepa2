from __future__ import annotations

import argparse

import torch
import torch.nn as nn


def _dinov2_frame_features(dino: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Return a single feature vector per image.

    Supports common Dinov2 APIs:
    - `forward_features(...)` returning a dict containing `x_norm_clstoken`
    - `forward(...)` returning a tensor [B, D]
    """
    if hasattr(dino, "forward_features"):
        feats = dino.forward_features(x)
        if isinstance(feats, dict):
            if "x_norm_clstoken" in feats:
                return feats["x_norm_clstoken"]
            if "x_clstoken" in feats:
                return feats["x_clstoken"]
        if torch.is_tensor(feats):
            return feats
    out = dino(x)
    if isinstance(out, dict):
        if "x_norm_clstoken" in out:
            return out["x_norm_clstoken"]
        if "x_clstoken" in out:
            return out["x_clstoken"]
    if torch.is_tensor(out):
        return out
    raise RuntimeError(f"Unsupported Dinov2 output type: {type(out)}")


class DinoFrameMLPClassifier(nn.Module):
    """
    (B, C, T, H, W) video -> Dinov2 per-frame features -> per-frame MLP -> meanpool over T -> linear classifier
    """

    def __init__(
        self,
        dino: nn.Module,
        dino_dim: int,
        mlp_hidden_dim: int,
        mlp_out_dim: int,
        num_classes: int,
        freeze_dino: bool = True,
    ):
        super().__init__()
        self.dino = dino
        self.freeze_dino = freeze_dino

        self.frame_mlp = nn.Sequential(
            nn.Linear(dino_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, mlp_out_dim),
        )
        self.head = nn.Linear(mlp_out_dim, num_classes)

        if self.freeze_dino:
            for p in self.dino.parameters():
                p.requires_grad_(False)
            self.dino.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep Dinov2 frozen in eval mode even when the rest of the model trains.
        if self.freeze_dino:
            self.dino.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected video tensor [B,C,T,H,W], got shape={tuple(x.shape)}")
        B, C, T, H, W = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, C, H, W)

        if self.freeze_dino:
            with torch.no_grad():
                f = _dinov2_frame_features(self.dino, frames)  # (B*T, D)
        else:
            f = _dinov2_frame_features(self.dino, frames)  # (B*T, D)

        f = f.reshape(B, T, -1)  # (B, T, D)
        f = self.frame_mlp(f)  # (B, T, M)
        pooled = f.mean(dim=1)  # (B, M)
        return self.head(pooled)


def _build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    # Load Dinov2 as an image encoder and apply it per-frame.
    try:
        dino = torch.hub.load(args.dino_repo, args.dino_model, pretrained=args.dino_pretrained)
    except Exception as e:
        raise RuntimeError(
            "Failed to load Dinov2 via torch.hub. "
            "If you're offline, make sure the Dinov2 hub weights are already cached or set --dino-pretrained false.\n"
            f"repo={args.dino_repo!r} model={args.dino_model!r} pretrained={args.dino_pretrained}\n"
            f"original_error={e}"
        )

    # Infer feature dimension with a tiny forward on CPU (cheap, avoids model-specific attribute assumptions).
    dino.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.crop_size, args.crop_size)
        d = int(_dinov2_frame_features(dino, dummy).shape[-1])

    model = DinoFrameMLPClassifier(
        dino=dino,
        dino_dim=d,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_out_dim=args.mlp_out_dim,
        num_classes=args.num_classes,
        freeze_dino=args.freeze_dino,
    )
    return model.to(device)


