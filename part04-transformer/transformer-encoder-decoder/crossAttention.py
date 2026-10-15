import math

import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dropout=0.0,
    ):
        super().__init__()
        
        if d_model % num_heads != 0:
            raise ValueError("d_model必须能被num_heads整除")
    
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x,
        memory,
        source_padding_mask=None,
        past_kv=None,
        use_cache=False,
    ):
        B, T, D = x.shape

        # 当前下联位置的 Q，每次都要计算。
        q = self.q_proj(x)
        q = q.reshape(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        if past_kv is None:
            # 上联 K、V 只在第一次计算。
            S = memory.size(1)

            k = self.k_proj(memory)
            v = self.v_proj(memory)

            k = k.reshape(
                B, S, self.num_heads, self.head_dim
            ).transpose(1, 2)

            v = v.reshape(
                B, S, self.num_heads, self.head_dim
            ).transpose(1, 2)
        else:
            k, v = past_kv

        S = k.size(-2)

        # [B, H, T, S]
        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        if source_padding_mask is not None:
            if source_padding_mask.dtype != torch.bool:
                raise TypeError("source_padding_mask 必须是 bool")

            if source_padding_mask.shape != (B, S):
                raise ValueError("source_padding_mask 必须为 [B, S]")

            if source_padding_mask.all(dim=-1).any():
                raise ValueError("每条上联至少需要一个有效 token")

            scores = scores.masked_fill(
                source_padding_mask[:, None, None, :],
                float("-inf"),
            )

        weights = torch.softmax(scores, dim=-1)
        out = self.attn_dropout(weights) @ v

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        if use_cache:
            return out, weights, (k, v)

        return out, weights