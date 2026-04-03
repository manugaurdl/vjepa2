"""
Causal transformer for next-frame prediction.
Processes all frames in parallel with causal attention mask.
Returns same 6-tuple interface as VideoRNNTransformerEncoder.
"""

import torch
import torch.nn as nn


class CausalTransformerPredictor(nn.Module):
    def __init__(self, dim, n_frames, n_patches=1, depth=4, n_heads=8):
        super().__init__()
        self.dim = dim
        self.n_frames = n_frames
        self.n_patches = n_patches
        self.seq_len = n_frames * n_patches

        self.frame_embed = nn.Embedding(n_frames, dim)
        if n_patches > 1:
            self.patch_embed = nn.Embedding(n_patches, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=4 * dim,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pred_head = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

        self._register_causal_mask()

    def _register_causal_mask(self):
        """Causal mask: attend to current and past frames. Patches within same frame can see each other."""
        S = self.seq_len
        P = self.n_patches
        mask = torch.ones(S, S, dtype=torch.bool)
        for i in range(S):
            frame_i = i // P
            for j in range(S):
                frame_j = j // P
                if frame_j <= frame_i:
                    mask[i, j] = False
        self.register_buffer("causal_mask", mask)

    def forward(self, x, state=None, return_all=True):
        """
        x: (B, T, D) for CLS or (B, T, S, D) for patches
        Returns same 6-tuple as VideoRNNTransformerEncoder.
        """
        squeeze_tokens = False
        if x.ndim == 3:
            B, T, D = x.shape
            S = 1
            squeeze_tokens = True
            tokens = x  # (B, T, D)
        else:
            B, T, S, D = x.shape
            tokens = x.reshape(B, T * S, D)

        # Positional embeddings
        frame_ids = torch.arange(T, device=x.device).unsqueeze(1).expand(T, S).reshape(-1)
        tokens = tokens + self.frame_embed(frame_ids)
        if S > 1:
            patch_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(T, S).reshape(-1)
            tokens = tokens + self.patch_embed(patch_ids)

        # Causal transformer
        tokens = self.transformer(tokens, mask=self.causal_mask)
        preds = self.pred_head(self.ln(tokens))  # (B, T*S, D)

        # Reshape
        if S > 1:
            preds = preds.reshape(B, T, S, D)
            targets = x
        else:
            preds = preds.reshape(B, T, D)
            targets = x

        # pred_error_l2: pred[t] should predict target[t+1]
        # Shape: (B, T-1) or (B, T-1, S)
        error_l2 = ((preds[:, :-1] - targets[:, 1:]) ** 2).sum(dim=-1)
        if error_l2.ndim == 2:
            error_l2 = error_l2.unsqueeze(-1)  # (B, T-1, 1) to match RNN's (B, T, S)

        # Pad with zeros at t=0 to get (B, T, S), matching RNN interface
        pad_shape = (B, 1, error_l2.shape[-1])
        pred_error_l2 = torch.cat([torch.zeros(pad_shape, device=x.device), error_l2], dim=1)

        # Final state: last frame's output
        if S > 1:
            outs = preds.reshape(B, T, S, D)
            final_state = outs[:, -1]  # (B, S, D)
        else:
            outs = preds
            final_state = outs[:, -1]  # (B, D)

        # Dummy values for RNN-specific diagnostics
        update_gates = torch.zeros(B, T, device=x.device)
        update_norms = torch.zeros(B, T, device=x.device)
        r_novelty = torch.zeros(B, T, device=x.device)

        return outs, final_state, update_gates, update_norms, r_novelty, pred_error_l2
