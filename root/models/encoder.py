from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn
from root.models.rnn import VideoRNNTransformerEncoder

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

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 4):
        super().__init__()
        self.out_dim = int(hidden_dim)
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2 (got {num_layers})")

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, self.out_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def build_encoder(encoder_cfg: Any, *, input_dim: int) -> Tuple[nn.Module, int]:
    """Returns: (encoder_module, encoder_out_dim)"""
    enc_type = encoder_cfg.type

    if enc_type == "mlp":
        mlp_cfg = encoder_cfg.mlp
        encoder = MLP(input_dim=input_dim, hidden_dim=mlp_cfg.hidden_dim, num_layers=mlp_cfg.num_layers)
        out_dim = encoder.out_dim
    elif enc_type == "linear":
        encoder = nn.Identity()
        out_dim = None
    elif enc_type == "transformer":
        tr_cfg = encoder_cfg.transformer
        encoder = Transformer(input_dim=input_dim, hidden_dim=tr_cfg.hidden_dim, depth=tr_cfg.depth, n_heads=tr_cfg.n_heads)
        out_dim = encoder.out_dim
    elif enc_type == "rnn":
        rnn_cfg = encoder_cfg.rnn
        encoder = VideoRNNTransformerEncoder(dim=rnn_cfg.hidden_dim, update_type=rnn_cfg.update_type, num_layers=rnn_cfg.depth, num_heads=8, mlp_dim=4*rnn_cfg.hidden_dim, cross_attn_dim=rnn_cfg.cross_attn_dim, decay_state=rnn_cfg.decay_state)
        out_dim = rnn_cfg.hidden_dim
    return encoder, out_dim
