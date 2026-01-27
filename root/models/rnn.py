import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidAttention(nn.Module):
    def __init__(
        self,
        dim: int,              # The internal attention dimension (num_heads * head_dim)
        dim_q: int = None,     # Input dimension of Query (defaults to dim if None)
        dim_kv: int = None,    # Input dimension of Key/Value (defaults to dim if None)
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        
        # Default to 'dim' if input dims are not specified
        dim_q = dim_q or dim
        dim_kv = dim_kv or dim

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 1. Projections accept different input dimensions (dim_q vs dim_kv)
        #    but project to the SAME internal dimension (dim)
        self.q_proj = nn.Linear(dim_q, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim_kv, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim_kv, dim, bias=qkv_bias)
        
        self.q_norm = norm_layer(self.head_dim)
        self.k_norm = norm_layer(self.head_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim_q) # Output projection usually goes back to 'dim'
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q: (B, Lq, dim_q)
        # kv: (B, Lk, dim_kv)
        
        B, Lq, _ = q.shape
        _, Lk, _ = k.shape

        q = self.q_proj(q).reshape(B, Lq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, Lk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, Lk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = q * self.scale
        attn = torch.sigmoid(q @ k.transpose(-2, -1)) 
        
        x = self.attn_drop(attn) @ v
        x = x.transpose(1, 2).reshape(B, Lq, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
        return x


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class CrossAttentionBlock(nn.Module):
    """
    Matches the ordering in your notebook's torch code:
      x = x + cross_attn(LN(x), kv)
      x = x + mlp(LN(x))
      x = x + self_attn(LN(x))
    """
    def __init__(self, dim: int, num_heads: int, mlp_dim: int, cross_attn_dim: int=None):
        super().__init__()
        self.ca_ln = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_ln = nn.LayerNorm(dim, eps=1e-6)
        self.sa_ln = nn.LayerNorm(dim, eps=1e-6)

        # self.sigmoid_ca = SigmoidAttention(dim=cross_attn_dim, dim_q=dim, dim_kv=dim, num_heads=num_heads)
        # self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.W_cross_attn = nn.Linear(dim, dim)
        self.W_self_attn = nn.Linear(dim, dim)
        # self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = FeedForward(dim, mlp_dim)

    def forward(self, x, kv):
        # cross-attn: queries from x, keys/values from kv
        q = self.ca_ln(x)
        k = kv
        v = kv
        # ca_out = self.sigmoid_ca(q, k, v) # sigmoid for cross-attention
        ca_out = self.W_cross_attn(kv) #replace CA with linear layer
        # ca_out, _ = self.cross_attn(q, k, v, need_weights=False)
        x = x + ca_out

        x = x + self.mlp(self.mlp_ln(x))

        qkv = self.sa_ln(x)
        sa_out = self.W_self_attn(qkv) # replace SA with linear layer
        # sa_out, _ = self.self_attn(qkv, qkv, qkv, need_weights=False)
        x = x + sa_out
        return x


class CrossAttentionTransformer(nn.Module):
    def __init__(self, num_layers: int, dim: int, num_heads: int, mlp_dim: int, cross_attn_dim: int=None):
        super().__init__()
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(dim, num_heads, mlp_dim, cross_attn_dim=cross_attn_dim) for _ in range(num_layers)]
        )
        self.out_ln = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x, kv):
        for blk in self.blocks:
            x = blk(x, kv)
        return self.out_ln(x)


class GatedTransformerCore(nn.Module):
    """
    GRU-like gating around a cross-attention transformer.
    Inputs and state are sequences of tokens: (B, S, D).
    """
    def __init__(self, dim: int, num_layers: int = 4, num_heads: int = 8, mlp_dim: int = None, cross_attn_dim: int=None):
        super().__init__()
        mlp_dim = (4 * dim) if mlp_dim is None else mlp_dim

        self.state_ln = nn.LayerNorm(dim, eps=1e-6)

        self.input_update = nn.Linear(dim, dim, bias=False)
        self.input_reset = nn.Linear(dim, dim, bias=False)
        self.state_update = nn.Linear(dim, dim, bias=False)
        self.state_reset = nn.Linear(dim, dim, bias=False)

        # self.transformer = CrossAttentionTransformer(
        #     num_layers=num_layers, dim=dim, num_heads=num_heads, mlp_dim=mlp_dim, cross_attn_dim=cross_attn_dim
        # )
        # self.W_input = nn.Linear(dim, dim, bias=False)
        # self.W_state = nn.Linear(dim, dim, bias=False)
        self.transformer = CrossAttentionTransformer(
            num_layers=num_layers, dim=dim, num_heads=num_heads, mlp_dim=mlp_dim, cross_attn_dim=cross_attn_dim
        )

    def forward(self, inputs, state):
        # inputs/state: (B, S, D)
        update_gate = torch.sigmoid(self.input_update(inputs) + self.state_update(state))
        
        ### reset gate
        # reset_gate = torch.sigmoid(self.input_reset(inputs) + self.state_reset(state))
        # kv = reset_gate * self.state_ln(state)
        ### no reset gate
        kv = self.state_ln(state)

        # h = torch.tanh(self.W_input(inputs) + self.W_state(kv)) # replace transformer with GRU linear layers
        h = self.transformer(inputs, kv)

        out = (1.0 - update_gate) * state + update_gate * h
        state = out
        return out, state


class VideoRNNTransformerEncoder(nn.Module):
    """
    Encodes a full video of frame features.

    Accepts:
      - (B, T, D) : one feature vector per frame
      - (B, T, S, D) : token sequences per frame (e.g., patches+CLS)

    Returns:
      - all_outputs: (B, T, D) or (B, T, S, D) if return_all=True
        else last_output: (B, D) or (B, S, D)
      - final_state: (B, 1, D) or (B, S, D)
    """
    def __init__(self, dim: int, num_layers: int = 4, num_heads: int = 8, mlp_dim: int = None, cross_attn_dim: int=None):
        super().__init__()
        self.core = GatedTransformerCore(
            dim=dim, num_layers=num_layers, num_heads=num_heads, mlp_dim=mlp_dim, cross_attn_dim=cross_attn_dim
        )

    def forward(self, x, state=None, return_all: bool = True):
        squeeze_tokens = False
        if x.dim() == 3:
            # (B, T, D) -> (B, T, S=1, D)
            x = x.unsqueeze(2)
            squeeze_tokens = True
        assert x.dim() == 4, f"Expected (B,T,D) or (B,T,S,D), got {tuple(x.shape)}"

        B, T, S, D = x.shape
        if state is None:
            state = torch.zeros((B, S, D), device=x.device, dtype=x.dtype)
 
        outs = []
        for t in range(T):
            out_t, state = self.core(x[:, t], state)  # (B,S,D)
            outs.append(out_t)

        outs = torch.stack(outs, dim=1)  # (B,T,S,D)

        if squeeze_tokens:
            outs = outs.squeeze(2)   # (B,T,D)
            state = state.squeeze(1) # (B,D)

        if return_all:
            return outs, state
        else:
            return outs[:, -1], state


# ---- minimal usage ----
# frame_feats: (B, T, D)
# encoder = VideoRNNTransformerEncoder(dim=D, num_layers=4, num_heads=8, mlp_dim=4*D).cuda()
# video_encodings, final_state = encoder(frame_feats.cuda(), return_all=True)
# last_encoding, final_state = encoder(frame_feats.cuda(), return_all=False)