from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from root.models.encoder import build_encoder


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

class DinoFrameEncoder(nn.Module):
    """
    (B, C, T, H, W) video -> Dinov2 per-frame features -> per-frame MLP -> meanpool over T -> linear classifier
    """

    def __init__(
        self,
        dino: nn.Module,
        dino_dim: int,
        args: argparse.Namespace,
        num_classes: int,
        pooling: str = "mean",
        freeze_dino: bool = True,
    ):
        super().__init__()
        self.dino = dino
        self.freeze_dino = freeze_dino
        self.pooling = pooling
        self.encoder_type = args.encoder.type
        self.cache_dino_feats = args.cache_dino_feats
        
        self.encoder, encoder_out_dim = build_encoder(args.encoder, input_dim=dino_dim)
        if encoder_out_dim is None:
            encoder_out_dim = dino_dim
        if self.encoder_type == "rnn":
            head_in_dim = args.encoder.rnn.hidden_dim
        else:
            if self.pooling == "concat":
                head_in_dim = encoder_out_dim * args.frames_per_clip
            elif self.pooling == "mean":
                head_in_dim = encoder_out_dim

        self.head = nn.Linear(head_in_dim, num_classes)

        if self.freeze_dino and (self.dino is not None):
            for p in self.dino.parameters():
                p.requires_grad_(False)
            self.dino.eval()
        
        if self.cache_dino_feats:
            self.id_to_feat = torch.zeros(168913, 8, 384) # train:168913, val: 24777
        if getattr(args, "val_dataset_len", None) is not None:
            self.update_gates = torch.zeros(args.val_dataset_len, args.eval_frames_per_clip)
            self.update_norms = torch.zeros(args.val_dataset_len, args.eval_frames_per_clip)
            self.r_novelty = torch.zeros(args.val_dataset_len, args.eval_frames_per_clip)
            self.hidden_states = torch.zeros(args.val_dataset_len, args.eval_frames_per_clip, 384)
            self.pred_error_l2s = torch.zeros(args.val_dataset_len, args.eval_frames_per_clip)
            self.collect_update_gates = False


    def train(self, mode: bool = True):
        super().train(mode)
        # Keep Dinov2 frozen in eval mode even when the rest of the model trains.
        if self.freeze_dino and (self.dino is not None):
            self.dino.eval()
        return self

    def forward(self, x: torch.Tensor, ds_index: int) -> torch.Tensor:

        if self.dino is None:
            frame_feats = x
        else:
            if x.ndim != 5:
                raise ValueError(f"Expected video tensor [B,C,T,H,W], got shape={tuple(x.shape)}")
            B, C, T, H, W = x.shape
            frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, C, H, W)
            if self.freeze_dino:
                with torch.no_grad():
                    f = _dinov2_frame_features(self.dino, frames)  # (B*T, D)
            else:
                f = _dinov2_frame_features(self.dino, frames)  # (B*T, D)

            frame_feats = f.reshape(B, T, -1)  # (B, T, D)
            
            if self.cache_dino_feats:
                self.id_to_feat[ds_index] = frame_feats.detach().cpu()
                return frame_feats

        frame_feats = self.encoder(frame_feats)  # (B, T, M)

        if self.encoder_type == "rnn":
            
            hidden_states, final_state, timesteps_update_gate, timesteps_update_norm, timesteps_r_novelty, pred_error_l2 = frame_feats ### hidden_states[:,-1] == final_state
            self.pred_error_l2 = pred_error_l2

            if (not self.training) and getattr(self, "collect_update_gates", False):
                self.update_gates[ds_index] = timesteps_update_gate.detach().cpu()
                self.hidden_states[ds_index] = hidden_states.detach().cpu()
                self.update_norms[ds_index] = timesteps_update_norm.detach().cpu()
                self.r_novelty[ds_index] = timesteps_r_novelty.detach().cpu()
                if pred_error_l2 is not None:
                    self.pred_error_l2s[ds_index] = pred_error_l2.mean(dim=-1).detach().cpu()
            return self.head(final_state)
        else:
            self.pred_error_l2 = None
            if self.pooling == "mean":
                pooled = frame_feats.mean(dim=1)
            elif self.pooling == "concat":
                pooled = frame_feats.reshape(B, -1)  # (B, T*M)
            return self.head(pooled)


def _build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.pooling == "concat" and (args.eval_frames_per_clip != args.frames_per_clip):
        raise ValueError("pooling='concat' requires frames_per_clip == eval_frames_per_clip (fixed T).")
    # Load Dinov2 as an image encoder and apply it per-frame.
    
    dino_dim = 384 if args.dino_model == "dinov2_vits14" else 1024
    if args.load_cache_feats:
        dino = None
    else:
        try:
            dino = torch.hub.load(args.dino_repo, args.dino_model, pretrained=args.dino_pretrained)
        except Exception as e:
            raise RuntimeError("Failed to load Dinov2 via torch.hub.")


    model = DinoFrameEncoder(
        dino=dino,
        dino_dim=dino_dim,
        args=args,
        num_classes=args.num_classes,
        pooling=args.pooling,
        freeze_dino=args.freeze_dino,
    )
    return model.to(device)


