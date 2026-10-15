import torch
import torch.nn as nn
import torch.nn.functional as F

from transformerBlock import TransformerBlock, LayerNorm

class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        max_seq_len,
        dropout = 0.0,
    ):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        
        #字符编号 -> 字符特征
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )
        
        # 序列位置0,1,2 ... ->位置特征
        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model,
        )
        
        self.embedding_dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model = d_model,
                    num_heads = num_heads,
                    max_seq_len = max_seq_len,
                    dropout = dropout,
                )
                for _ in range(num_layers)
            ]
        )
        
        self.final_norm = LayerNorm(d_model)
        
        # 每个位置输出整个字符表的分数
        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias = False,
        )
    
    def forward(
        self,
        token_ids,
        targets = None,
        past_key_values = None,
        use_cache = False,
    ):
        B, T = token_ids.shape
        
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].shape[-2]
            
        total_length = past_length + T
        
        if total_length > self.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds "
                f"max_seq_len {self.max_seq_len}"
            )
            
        #有cache时,新的token必须使用其真实位置
        positions = torch.arange(
            past_length,
            total_length,
            device = token_ids.device,
        )
        
        token_features = self.token_embedding(token_ids)
        position_features = self.position_embedding(positions)
        
        #position_features是[T, D]
        #PyTorch会自动广播到batch的每一个样本
        x = token_features + position_features
        x = self.embedding_dropout(x)
        
        present_key_values = []
        
        for layer_index, block in enumerate(self.blocks):
            layer_past_kv = None
            
            if past_key_values is not None:
                layer_past_kv = past_key_values[layer_index]
            
            if use_cache:
                x, layer_present_kv = block(
                    x,
                    past_kv = layer_past_kv,
                    use_cache = True,
                )

                present_key_values.append(layer_present_kv)
            else:
                x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(B * T, -1),
                targets.reshape(B * T),
            )
        
        if use_cache:
            return logits, loss, present_key_values
        
        return logits, loss
    
    @torch.no_grad()
    def generate_greedy(
        self,
        token_ids,
        max_new_tokens
    ):
        self.eval()
        
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.max_seq_len:]
            
            logits, _ = self(context)
            
            #生成时只需要最后一个位置的预测
            next_token_logits = logits[:, -1, :]
            
            next_token = torch.argmax(
                next_token_logits,
                dim = -1,
                keepdim = True
            )
            
            token_ids = torch.cat(
                [token_ids, next_token],
                dim = 1,
            )
            
        return token_ids
    
def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 重复语料的目的只是验证模型可以记住一个确定规律。
    text = "hello transformer\n" * 200

    chars = sorted(set(text))
    vocab_size = len(chars)

    char_to_id = {
        char: index
        for index, char in enumerate(chars)
    }

    id_to_char = {
        index: char
        for char, index in char_to_id.items()
    }

    def encode(s):
        return [char_to_id[char] for char in s]

    def decode(ids):
        return "".join(id_to_char[index] for index in ids)

    data = torch.tensor(
        encode(text),
        dtype=torch.long,
    )

    max_seq_len = 32
    batch_size = 32

    def get_batch():
        starts = torch.randint(
            low=0,
            high=len(data) - max_seq_len,
            size=(batch_size,),
        )

        inputs = torch.stack(
            [
                data[start:start + max_seq_len]
                for start in starts
            ]
        )

        # target 相对 input 向右移动一个字符。
        targets = torch.stack(
            [
                data[start + 1:start + max_seq_len + 1]
                for start in starts
            ]
        )

        return inputs.to(device), targets.to(device)

    model = TinyGPT(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=3,
        max_seq_len=max_seq_len,
        dropout=0.0,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("device:", device)
    print("vocabulary:", repr("".join(chars)))
    print("parameters:", parameter_count)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
        weight_decay=0.0,
    )

    model.train()

    for step in range(1001):
        inputs, targets = get_batch()

        _, loss = model(inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Transformer 训练中保留梯度裁剪是合理的保险措施。
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        if step % 100 == 0:
            print(
                f"step {step:4d} | "
                f"loss {loss.item():.6f}"
            )

    prompt = "hello"

    prompt_ids = torch.tensor(
        [encode(prompt)],
        dtype=torch.long,
        device=device,
    )

    generated_ids = model.generate_greedy(
        prompt_ids,
        max_new_tokens=80,
    )

    generated_text = decode(
        generated_ids[0].tolist()
    )

    print("\ngenerated:")
    print(repr(generated_text))


if __name__ == "__main__":
    main()
            