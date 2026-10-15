import torch

from tinyGPT import TinyGPT


torch.manual_seed(42)

# 使用 CPU，避免 GPU 不同矩阵乘法带来的额外数值误差。
device = torch.device("cpu")

model = TinyGPT(
    vocab_size=20,
    d_model=32,
    num_heads=4,
    num_layers=2,
    max_seq_len=16,
    dropout=0.0,
).to(device)

# 必须关闭 dropout。
model.eval()

# 假设这是由 7 个字符组成的序列。
tokens = torch.tensor(
    [[3, 7, 2, 9, 4, 6, 1]],
    dtype=torch.long,
    device=device,
)


# ============================================================
# 方法一：像训练一样，一次输入完整序列
# ============================================================

with torch.no_grad():
    full_logits, _ = model(tokens)

print("完整重算完成")


# ============================================================
# 方法二：模拟生成，每次只输入一个 token
# ============================================================

cache = None
cached_logits_list = []

with torch.no_grad():
    for position in range(tokens.shape[1]):
        current_token = tokens[
            :,
            position:position + 1,
        ]

        current_logits, _, cache = model(
            current_token,
            past_key_values=cache,
            use_cache=True,
        )

        cached_logits_list.append(current_logits)

        cache_length = cache[0][0].shape[-2]

        print(
            f"输入 token {position}: "
            f"{current_token.item()}, "
            f"cache 长度: {cache_length}"
        )


# 把每一步得到的 logits 拼成完整序列。
cached_logits = torch.cat(
    cached_logits_list,
    dim=1,
)


# ============================================================
# 比较两条计算路径
# ============================================================

difference = (
    full_logits - cached_logits
).abs()

print()
print("最大 logits 差异:", difference.max().item())

print(
    "是否数值等价:",
    torch.allclose(
        full_logits,
        cached_logits,
        atol=1e-5,
        rtol=1e-5,
    ),
)