import torch
import torch.nn as nn

from crossAttention import CrossAttention
from transformerBlock import LayerNorm, FeedForward


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dropout=0.0,
    ):
        super().__init__()

        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

        # 同一份输入同时提供 Q、K、V，因此执行 Self-Attention。
        self.attention = CrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.attention_out_dropout = nn.Dropout(dropout)

        self.ffn = FeedForward(
            d_model=d_model,
            dropout=dropout,
        )

    def forward(self, x, source_padding_mask=None):
        """
        x:                   [B, S, D]
        source_padding_mask: [B, S]，True 表示 PAD
        """
        normalized = self.ln1(x)

        # Q、K、V 都由 normalized 产生。
        attention_output, _ = self.attention(
            normalized,
            normalized,
            source_padding_mask=source_padding_mask,
        )

        x = x + self.attention_out_dropout(attention_output)
        x = x + self.ffn(self.ln2(x))

        return x