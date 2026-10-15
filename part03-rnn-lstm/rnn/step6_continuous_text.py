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
# 1. 连续文本
# --------------------------------------------------

torch.manual_seed(42)

text = "hello rnn hello rnn"

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

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

input_text = text[:-1]
target_text = text[1:]

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

# --------------------------------------------------
# 2. 创建模型
# --------------------------------------------------

hidden_size = 16
chunk_len = 4
num_epochs = 100

model = CharRNN(
    vocab_size = vocab_size,
    hidden_size = hidden_size,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.03,
)

# --------------------------------------------------
# 3. 截断 BPTT 训练
# --------------------------------------------------

total_steps = input_ids.shape[1]

for epoch in range(1, num_epochs + 1):
    model.train()
    
    hidden = None
    
    epoch_loss_sum = 0.0
    epoch_correct = 0
    epoch_tokens = 0
    
    for chunk_index, start in enumerate(
        range(0, total_steps, chunk_len)
    ):
        end = min(start + chunk_len, total_steps)
        
        chunk_input = input_ids[:, start:end]
        chunk_target = target_ids[:, start:end]
        
        if hidden is not None:
            hidden_before_detach = hidden
            hidden = hidden.detach()

            # detach 不应改变数值
            assert torch.equal(
                hidden,
                hidden_before_detach,
            )

            # detach 后不再连接旧计算图
            assert hidden.grad_fn is None
            assert hidden.requires_grad is False

            if epoch == 1:
                print(
                    f"chunk {chunk_index}: "
                    "隐藏状态数值已传递，"
                    "旧计算图已切断"
                )

        logits, hidden = model(
            chunk_input,
            initial_hidden=hidden,
        )

        # 新片段经过 RNN 后，又建立了当前片段的计算图
        assert hidden.grad_fn is not None
        assert hidden.requires_grad is True

        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            chunk_target.reshape(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        with torch.no_grad():
            predicted_ids = logits.argmax(dim=-1)
            correct = (
                predicted_ids == chunk_target
            ).sum().item()

        number_of_tokens = chunk_target.numel()

        epoch_loss_sum += (
            loss.item() * number_of_tokens
        )
        epoch_correct += correct
        epoch_tokens += number_of_tokens

        if epoch == 1:
            print(
                f"  范围=[{start}, {end}), "
                f"长度={end - start}, "
                f"hidden_norm="
                f"{hidden.detach().norm().item():.6f}"
            )

    average_loss = epoch_loss_sum / epoch_tokens
    accuracy = epoch_correct / epoch_tokens

    if epoch == 1 or epoch % 20 == 0:
        print(
            f"epoch={epoch:3d}, "
            f"loss={average_loss:.6f}, "
            f"accuracy={accuracy:.2%}"
        )

    if accuracy == 1.0 and average_loss < 0.02:
        print(
            f"epoch={epoch:3d}, "
            f"loss={average_loss:.6f}, "
            f"accuracy={accuracy:.2%}"
        )
        print("达到过拟合目标，提前停止。")
        break

# --------------------------------------------------
# 4. 按相同顺序进行 teacher-forcing 推理
# --------------------------------------------------

model.eval()

hidden = None
all_predicted_ids = []

with torch.no_grad():
    for start in range(0, total_steps, chunk_len):
        end = min(start + chunk_len, total_steps)

        chunk_input = input_ids[:, start:end]

        logits, hidden = model(
            chunk_input,
            initial_hidden=hidden,
        )

        chunk_predictions = logits.argmax(dim=-1)
        all_predicted_ids.append(chunk_predictions)

predicted_ids = torch.cat(
    all_predicted_ids,
    dim=1,
)

predicted_text = "".join(
    idx_to_char[index.item()]
    for index in predicted_ids[0]
)

print()
print("输入文本：", repr(input_text))
print("正确目标：", repr(target_text))
print("模型预测：", repr(predicted_text))

assert predicted_ids.shape == target_ids.shape

def generate_text(
    model,
    prompt,
    num_new_chars,
    char_to_idx,
    idx_to_char,
    temperature=0.0,
):
    """
    prompt:
        起始文本，例如 "h"

    num_new_chars:
        在 prompt 后继续生成多少个字符

    temperature:
        0.0 表示贪心生成
        > 0 表示按 temperature 随机采样

    return:
        包含 prompt 的完整字符串
    """

    if len(prompt) == 0:
        raise ValueError("prompt 不能为空")

    for char in prompt:
        if char not in char_to_idx:
            raise ValueError(
                f"字符 {char!r} 不在字符表中"
            )

    if temperature < 0:
        raise ValueError(
            "temperature 不能小于 0"
        )

    device = next(model.parameters()).device

    model.eval()

    generated_chars = list(prompt)
    hidden = None
    logits = None

    with torch.no_grad():
        # --------------------------------------------------
        # 1. 依次读入 prompt，建立对应的隐藏状态
        # --------------------------------------------------

        for char in prompt:
            input_id = torch.tensor(
                [[char_to_idx[char]]],
                dtype=torch.long,
                device=device,
            )

            logits, hidden = model(
                input_id,
                initial_hidden=hidden,
            )

        # 此时最后一个 logits 对应：
        # 看完整个 prompt 后，对下一个字符的预测

        # --------------------------------------------------
        # 2. 自回归生成
        # --------------------------------------------------

        for _ in range(num_new_chars):
            next_logits = logits[:, -1, :]

            if temperature == 0.0:
                # 贪心：选择 logits 最大的字符
                next_id = next_logits.argmax(
                    dim=-1
                )
            else:
                scaled_logits = (
                    next_logits / temperature
                )

                probabilities = torch.softmax(
                    scaled_logits,
                    dim=-1,
                )

                next_id = torch.multinomial(
                    probabilities,
                    num_samples=1,
                ).squeeze(dim=-1)

            next_char = idx_to_char[
                next_id.item()
            ]

            generated_chars.append(next_char)

            print(
                f"生成第 {len(generated_chars) - len(prompt):2d} 个字符："
                f"{next_char!r}"
            )
            
            # 把刚预测的字符作为下一时间步输入
            next_input = next_id.reshape(1, 1)

            logits, hidden = model(
                next_input,
                initial_hidden=hidden,
            )

    return "".join(generated_chars)

generated = generate_text(
    model=model,
    prompt="h",
    num_new_chars=len(text) - 1,
    char_to_idx=char_to_idx,
    idx_to_char=idx_to_char,
    temperature=0.0,
)

print()
print("训练文本：", repr(text))
print("生成文本：", repr(generated))
        