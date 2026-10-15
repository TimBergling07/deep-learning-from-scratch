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

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.rnn_cell = VanillaRNNCell(
            input_size = vocab_size,
            hidden_size = hidden_size,
        )
        
        #隐藏状态到字符logits
        self.W_hy = nn.Parameter(
            torch.empty(hidden_size, vocab_size)
        )
        self.b_y = nn.Parameter(
            torch.zeros(vocab_size)
        )
        
        nn.init.xavier_uniform_(self.W_hy)
    
    def forward(self, input_ids, initial_hidden = None):
        """
        input_ids:
            [B, T]

        initial_hidden:
            [B, H] 或 None

        return:
            logits:     [B, T, V]
            last_hidden:[B, H]
        """
        batch_size, seq_len = input_ids.shape
        
        #[B, T] -> [B, T, V]
        x = F.one_hot(
            input_ids,
            num_classes = self.vocab_size,
        ).to(torch.float32)
        
        if initial_hidden is None:
            h_t = torch.zeros(
                batch_size,
                self.hidden_size,
                device = input_ids.device,
            )
        else:
            h_t = initial_hidden
        
        logits_at_each_step = []
        
        for t in range(seq_len):
            #[B, V]
            x_t = x[:, t, :]
            
            #[B, H]
            h_t = self.rnn_cell(x_t, h_t)
            
            #[B, H] @ [H, V] -> [B, V]
            logits_t = h_t @ self.W_hy + self.b_y
            
            logits_at_each_step.append(logits_t)
            
        #T个[B, V] -> [B, T, V]
        logits = torch.stack(
            logits_at_each_step,
            dim = 1,
        )
        
        return logits, h_t
            
# --------------------------------------------------
# 1. 准备文本
# --------------------------------------------------

torch.manual_seed(42)

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

vocab_size = len(chars)

input_text = text[:-1]
target_text = text[1:]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

input_ids = torch.tensor(
    [[char_to_idx[char] for char in input_text]],
    dtype=torch.long,
    device=device,
)

target_ids = torch.tensor(
    [[char_to_idx[char] for char in target_text]],
    dtype=torch.long,
    device=device,
)

print("device：", device)
print("字符表：", chars)
print("输入：", repr(input_text))
print("目标：", repr(target_text))
print("input_ids.shape：", input_ids.shape)
print("target_ids.shape：", target_ids.shape)


# --------------------------------------------------
# 2. 创建模型
# --------------------------------------------------

hidden_size = 16

model = CharRNN(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.03,
)


# --------------------------------------------------
# 3. 在单个样本上训练
# --------------------------------------------------

def compute_global_grad_norm(parameters):
    """
    计算所有参数梯度合在一起的全局L2范数
    """
    
    total_squared_norm = None
    
    for parameter in parameters:
        if parameter.grad is None:
            continue

        squared_norm = parameter.grad.detach().pow(2).sum()
        
        if total_squared_norm is None:
            total_squared_norm = squared_norm
        else:
            total_squared_norm += squared_norm
    
    if total_squared_norm is None:
        return 0.0
    
    return total_squared_norm.sqrt().item()

max_epochs = 1000

for epoch in range(1, max_epochs + 1):
    model.train()

    logits, last_hidden = model(input_ids)

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    #裁剪之前的全局梯度范数
    grad_norm_before = compute_global_grad_norm(
        model.parameters()
    )
    
    max_grad_norm = 0.25
    
    returned_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm = max_grad_norm,
    )
    
    grad_norm_after = compute_global_grad_norm(
        model.parameters()
    )
    
    assert torch.isfinite(
        torch.tensor(grad_norm_before)
    )

    assert torch.isfinite(
        torch.tensor(grad_norm_after)
    )

    assert grad_norm_after <= max_grad_norm + 1e-5

    # clip_grad_norm_ 返回的也是裁剪前范数
    assert abs(
        returned_norm.item() - grad_norm_before
    ) < 1e-4
        
    optimizer.step()

    with torch.no_grad():
        predicted_ids = logits.argmax(dim=-1)

        accuracy = (
            predicted_ids == target_ids
        ).float().mean().item()

    if epoch == 1 or epoch % 20 == 0:
        was_clipped = grad_norm_before > max_grad_norm

        print(
            f"epoch={epoch:4d}, "
            f"loss={loss.item():.6f}, "
            f"accuracy={accuracy:.2%}, "
            f"grad={grad_norm_before:.6f} -> "
            f"{grad_norm_after:.6f}, "
            f"clipped={was_clipped}"
        )   

    # 单样本正确性测试：已经完全记住后即可停止
    if accuracy == 1.0 and loss.item() < 0.02:
        print(
            f"epoch={epoch:4d}, "
            f"loss={loss.item():.6f}, "
            f"accuracy={accuracy:.2%}, "
            f"grad={grad_norm_before:.6f} -> "
            f"{grad_norm_after:.6f}"
        )
        print("达到过拟合目标，提前停止。")
        break


# --------------------------------------------------
# 4. 最终检查
# --------------------------------------------------

model.eval()

with torch.no_grad():
    logits, last_hidden = model(input_ids)
    predicted_ids = logits.argmax(dim=-1)

predicted_text = "".join(
    idx_to_char[index.item()]
    for index in predicted_ids[0]
)

print()
print("输入文本：", repr(input_text))
print("正确目标：", repr(target_text))
print("模型预测：", repr(predicted_text))

print()
print("logits.shape：", logits.shape)
print("last_hidden.shape：", last_hidden.shape)

assert logits.shape == (
    1,
    len(input_text),
    vocab_size,
)

assert last_hidden.shape == (
    1,
    hidden_size,
)

assert torch.isfinite(logits).all()

print("尺寸与有限值检查通过。")

        