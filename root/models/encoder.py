from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn


class Transformer(nn.Module):
    """
    Token-wise transformer encoder for per-frame features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        n_heads: int,
        output_dim: int | None = None,
    ):
        super().__init__()
        # Lazy import so users can run with encoder.type="mlp" without having x_transformers installed.
        from x_transformers import Encoder  # type: ignore

        self.in_proj = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        self.encoder = Encoder(dim=hidden_dim, depth=depth, heads=n_heads, attn_dim_head = (hidden_dim // n_heads))
        self.out_dim = int(output_dim) if output_dim is not None else int(hidden_dim)
        # self.out_proj = nn.Identity() if self.out_dim == hidden_dim else nn.Linear(hidden_dim, self.out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        return self.encoder(x)
        # return self.out_proj(x)


class MLP(nn.Module):
    """
    Token-wise MLP applied independently to each time-step (frame feature).

    Works on tensors (B, T, D) because `nn.Linear` applies over the last dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.out_dim = int(output_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.out_dim),
        )## this is inverted ; latent should be wider. input/output dim should be same right?

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def build_encoder(encoder_cfg: Any, *, input_dim: int) -> Tuple[nn.Module, int]:
    """Returns: (encoder_module, encoder_out_dim)"""
    enc_type = encoder_cfg.type

    if enc_type == "mlp":
        mlp_cfg = encoder_cfg.mlp
        encoder = MLP(input_dim=input_dim, hidden_dim=mlp_cfg.hidden_dim, output_dim=mlp_cfg.output_dim)
        out_dim = mlp_cfg.output_dim
    elif enc_type == "linear":
        encoder = nn.Identity()
        out_dim = None
    elif enc_type == "transformer":
        tr_cfg = encoder_cfg.transformer
        encoder = Transformer(input_dim=input_dim, hidden_dim=tr_cfg.hidden_dim, depth=tr_cfg.depth, n_heads=tr_cfg.n_heads)
        out_dim = encoder.out_dim
        
    return encoder, out_dim
