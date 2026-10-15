import torch
import torch.nn as nn

from transformerBlock import TransformerBlock, LayerNorm
from crossAttention import CrossAttention


class DecoderBlock(TransformerBlock):
    def __init__(
        self,
        d_model,
        num_heads,
        max_seq_len,
        dropout=0.0,
    ):
        # 复用父类创建的：
        # self.ln1、self.ln2、self.attention、self.ffn
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Cross-Attention 使用独立的 LayerNorm。
        self.ln_cross = LayerNorm(d_model)

        self.cross_attention = CrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 之前的 CrossAttention 没有输出 dropout，在这里补上。
        self.cross_out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        memory,
        source_padding_mask=None,
        past_kv=None,
        use_cache=False,
    ):
        # 每层缓存包含两部分：
        # ((self_k, self_v), (cross_k, cross_v))
        if past_kv is None:
            past_self_kv = None
            past_cross_kv = None
        else:
            past_self_kv, past_cross_kv = past_kv

        # 1. 下联 Self-Attention。
        self_result = self.attention(
            self.ln1(x),
            past_kv=past_self_kv,
            use_cache=use_cache,
        )

        if use_cache:
            self_output, present_self_kv = self_result
        else:
            self_output = self_result

        x = x + self_output

        # 2. 读取上联，复用上联 K、V。
        cross_result = self.cross_attention(
            self.ln_cross(x),
            memory,
            source_padding_mask=source_padding_mask,
            past_kv=past_cross_kv,
            use_cache=use_cache,
        )

        if use_cache:
            cross_output, _, present_cross_kv = cross_result
        else:
            cross_output, _ = cross_result

        x = x + self.cross_out_dropout(cross_output)

        # 3. FFN。
        x = x + self.ffn(self.ln2(x))

        if use_cache:
            return x, (present_self_kv, present_cross_kv)

        return x