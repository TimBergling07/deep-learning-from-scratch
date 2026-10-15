import torch
import torch.nn as nn

from encoderBlock import EncoderBlock
from decoderBlock import DecoderBlock
from transformerBlock import LayerNorm


class CoupletTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        d_model=384,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        max_seq_len=72,
        dropout=0.15,
    ):
        super().__init__()

        self.pad_id = pad_id
        self.max_seq_len = max_seq_len

        # 上下联共用字符 embedding。
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id,
        )

        # 两边各自使用位置 embedding。
        self.source_position_embedding = nn.Embedding(
            max_seq_len,
            d_model,
        )
        self.target_position_embedding = nn.Embedding(
            max_seq_len,
            d_model,
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_encoder_layers)
        ])

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        ])

        # Pre-Norm 堆叠之后，再做一次最终归一化。
        self.encoder_norm = LayerNorm(d_model)
        self.decoder_norm = LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

    def embed(
        self,
        token_ids,
        position_embedding,
        start_position=0,
    ):
        length = token_ids.size(1)
        end_position = start_position + length

        if end_position > self.max_seq_len:
            raise ValueError(
                f"位置范围超过 max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(
            start_position,
            end_position,
            device=token_ids.device,
        )

        return self.embedding_dropout(
            self.token_embedding(token_ids)
            + position_embedding(positions)
        )

    def encode(self, source_ids, source_padding_mask=None):
        """
        source_ids: [B, S]
        返回 memory: [B, S, D]
        """
        if source_padding_mask is None:
            source_padding_mask = source_ids.eq(self.pad_id)

        x = self.embed(
            source_ids,
            self.source_position_embedding,
        )

        for block in self.encoder_blocks:
            x = block(
                x,
                source_padding_mask=source_padding_mask,
            )

        return self.encoder_norm(x)

    def decode(
        self,
        target_ids,
        memory,
        source_padding_mask,
        past_key_values=None,
        use_cache=False,
    ):
        if past_key_values is None:
            past_length = 0
        else:
            if len(past_key_values) != len(self.decoder_blocks):
                raise ValueError("缓存层数与 Decoder 层数不一致")

            # 第一层的下联 K 长度，就是已处理的位置数。
            first_self_kv, _ = past_key_values[0]
            past_length = first_self_kv[0].size(-2)

        x = self.embed(
            target_ids,
            self.target_position_embedding,
            start_position=past_length,
        )

        present_key_values = []

        for index, block in enumerate(self.decoder_blocks):
            layer_past = (
                None
                if past_key_values is None
                else past_key_values[index]
            )

            result = block(
                x,
                memory,
                source_padding_mask=source_padding_mask,
                past_kv=layer_past,
                use_cache=use_cache,
            )

            if use_cache:
                x, layer_present = result
                present_key_values.append(layer_present)
            else:
                x = result

        logits = self.lm_head(self.decoder_norm(x))

        if use_cache:
            return logits, present_key_values

        return logits

    def forward(self, source_ids, target_ids):
        """
        source_ids: 上联 token，右侧补 PAD
        target_ids: 下联输入 token，右侧补 PAD
        """
        source_padding_mask = source_ids.eq(self.pad_id)

        memory = self.encode(
            source_ids,
            source_padding_mask=source_padding_mask,
        )

        logits = self.decode(
            target_ids,
            memory,
            source_padding_mask,
        )

        return logits