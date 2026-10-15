import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_xh = nn.Parameter(
            torch.empty(input_size, hidden_size)
        )
        self.W_hh = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.b_h = nn.Parameter(
            torch.zeros(hidden_size)
        )

        nn.init.xavier_uniform_(self.W_xh)
        nn.init.xavier_uniform_(self.W_hh)

    def forward(self, x_t, h_prev):
        """
        x_t:    [B, V]
        h_prev: [B, H]
        return: [B, H]
        """
        h_t = torch.tanh(
            x_t @ self.W_xh
            + h_prev @ self.W_hh
            + self.b_h
        )

        return h_t

class CharPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        
        #[H, V]
        self.W_hy = nn.Parameter(
            torch.empty(hidden_size, vocab_size)
        )
        
        #[V]
        self.b_y = nn.Parameter(
            torch.zeros(vocab_size)
        )
        
        nn.init.xavier_uniform_(self.W_hy)
    
    def forward(self, hidden):
        """
        hidden:
            单时间步时可以是 [B, H]
            整个序列时可以是 [B, T, H]

        return:
            单时间步：[B, V]
            整个序列：[B, T, V]
        """
        logits = hidden @ self.W_hy + self.b_y
        return logits
    
torch.manual_seed(42)

# --------------------------------------------------
# 1. 字符表和训练样本
# --------------------------------------------------

text = "hello rnn"

chars = sorted(set(text))

char_to_idx = {
    char: index
    for index, char in enumerate(chars)
}

idx_to_char = {
    index: char
    for char, index in char_to_idx.items()
}

input_chars = "hell"
target_chars = "ello"

input_ids = torch.tensor(
    [char_to_idx[char] for char in input_chars],
    dtype=torch.long,
).unsqueeze(0) #unsqueeze 在第0维增加一个维度，相当于增加一个batch维度

target_ids = torch.tensor(
    [char_to_idx[char] for char in target_chars],
    dtype=torch.long,
).unsqueeze(0)

print("输入字符：", repr(input_chars))
print("目标字符：", repr(target_chars))
print("input_ids.shape：", input_ids.shape)
print("target_ids.shape：", target_ids.shape)


# --------------------------------------------------
# 2. 将整个输入序列转换为 one-hot
# --------------------------------------------------

vocab_size = len(chars)

x = F.one_hot(
    input_ids,
    num_classes=vocab_size,
).to(torch.float32)

print()
print("x.shape：", x.shape)

# x 的尺寸为 [B, T, V]
batch_size, seq_len, input_size = x.shape

print("batch_size =", batch_size)
print("seq_len =", seq_len)
print("input_size =", input_size)


# --------------------------------------------------
# 3. 创建 Cell 和初始隐藏状态
# --------------------------------------------------

hidden_size = 8

cell = VanillaRNNCell(
    input_size=input_size,
    hidden_size=hidden_size,
)

h_t = torch.zeros(batch_size, hidden_size)

hidden_states = []
parameter_addresses = []


# --------------------------------------------------
# 4. 沿时间维度展开
# --------------------------------------------------

for t in range(seq_len):
    # 从 [B, T, V] 中取出第 t 个时间步
    # x_t 的尺寸为 [B, V]
    x_t = x[:, t, :]

    h_t = cell(x_t, h_t)

    # h_t是非叶子张量，显示要求pytorch保留他的梯度
    h_t.retain_grad()
    
    hidden_states.append(h_t)

    # 记录参数底层存储地址，用来验证参数共享
    parameter_addresses.append(cell.W_xh.data_ptr())

    current_char = input_chars[t]

    print()
    print(f"时间步 t={t}")
    print("当前字符：", repr(current_char))
    print("x_t.shape：", x_t.shape)
    print("h_t.shape：", h_t.shape)
    print("h_t：", h_t)


# --------------------------------------------------
# 5. 合并所有时间步的隐藏状态
# --------------------------------------------------

all_hidden = torch.stack(hidden_states, dim=1)

print()
print("all_hidden.shape：", all_hidden.shape)
print("参数地址：", parameter_addresses)


# --------------------------------------------------
# 6. 验证
# --------------------------------------------------

assert input_ids.shape == (batch_size, seq_len)
assert x.shape == (batch_size, seq_len, vocab_size)
assert all_hidden.shape == (batch_size, seq_len, hidden_size)

# 每个时间步记录的参数地址必须相同
assert len(set(parameter_addresses)) == 1

print()
print("参数共享验证通过。")
print("所有尺寸检查通过。")

# --------------------------------------------------
# 7. 将隐藏状态映射为字符 logits
# --------------------------------------------------

prediction_head = CharPredictionHead(
    hidden_size = hidden_size,
    vocab_size = vocab_size,
)

logits = prediction_head(all_hidden)

print()
print("all_hidden.shape：", all_hidden.shape)
print("W_hy.shape：", prediction_head.W_hy.shape)
print("b_y.shape：", prediction_head.b_y.shape)
print("logits.shape：", logits.shape)

assert logits.shape == (
    batch_size,
    seq_len,
    vocab_size,
)

# --------------------------------------------------
# 8. 计算所有时间步的交叉熵损失
# --------------------------------------------------

# cross_entropy 要求：
#   输入：[样本数量, 类别数量]
#   目标：[样本数量]
#
# 因此把 B 和 T 合并成一个维度。

flat_logits = logits.reshape(
    batch_size * seq_len, 
    vocab_size,
)

flat_targets = target_ids.reshape(
    batch_size * seq_len
)

loss = F.cross_entropy(
    flat_logits,
    flat_targets,
)

print()
print("flat_logits.shape：", flat_logits.shape)
print("flat_targets.shape：", flat_targets.shape)
print("loss：", loss.item())

# --------------------------------------------------
# 9. 查看未训练模型的预测
# --------------------------------------------------

predicted_ids = logits.argmax(dim=-1)

predicted_text = "".join(
    idx_to_char[index.item()]
    for index in predicted_ids[0]
)

print()
print("输入文本：", repr(input_chars))
print("正确目标：", repr(target_chars))
print("当前预测：", repr(predicted_text))

assert predicted_ids.shape == target_ids.shape

print()
print("前向传播和损失计算检查通过。")

# --------------------------------------------------
# 10. 执行一次 BPTT
# --------------------------------------------------

cell.zero_grad(set_to_none = True)
prediction_head.zero_grad(set_to_none = True)

loss.backward()

print()
print("BPTT完成")

print()
print("参数梯度尺寸：")

print("W_xh.grad.shape：", cell.W_xh.grad.shape)
print("W_hh.grad.shape：", cell.W_hh.grad.shape)
print("b_h.grad.shape：", cell.b_h.grad.shape)

print("W_hy.grad.shape：", prediction_head.W_hy.grad.shape)
print("b_y.grad.shape：", prediction_head.b_y.grad.shape)

print()
print("参数梯度范数：")

print(
    "||grad W_xh|| =",
    cell.W_xh.grad.norm().item()
)

print(
    "||grad W_hh|| =",
    cell.W_hh.grad.norm().item()
)

print(
    "||grad b_h||  =",
    cell.b_h.grad.norm().item()
)

print(
    "||grad W_hy|| =",
    prediction_head.W_hy.grad.norm().item()
)

print(
    "||grad b_y||  =",
    prediction_head.b_y.grad.norm().item()
)

parameters = {
    "W_xh": cell.W_xh,
    "W_hh": cell.W_hh,
    "b_h": cell.b_h,
    "W_hy": prediction_head.W_hy,
    "b_y": prediction_head.b_y,
}

for name, parameter in parameters.items():
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()

    print(
        f"{name:4s}: "
        f"shape={tuple(parameter.grad.shape)}, "
        f"norm={parameter.grad.norm().item():.6f}"
    )
    
print()
print("各时间步隐藏状态的梯度范数：")

for t, hidden in enumerate(hidden_states):
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()

    print(
        f"t={t}, "
        f"字符={input_chars[t]!r}, "
        f"||dL/dh_t||={hidden.grad.norm().item():.6f}"
    )