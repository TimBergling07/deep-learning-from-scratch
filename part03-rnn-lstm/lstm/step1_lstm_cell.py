import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCellFromScratch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #一次计算输入门，遗忘门，候选记忆，输出门
        # W_x:[V, 4H]
        # W_h:[H, 4H]
        # b: [4H]
        self.W_x = nn.Parameter(
            torch.empty(
                input_size,
                4 * hidden_size,
            )
        )
        
        self.W_h = nn.Parameter(
            torch.empty(
                hidden_size,
                4 * hidden_size,
            )
        )
        
        self.b = nn.Parameter(
            torch.zeros(4 * hidden_size)
        )
        
        nn.init.xavier_uniform(self.W_x)
        nn.init.xavier_uniform(self.W_h)
        
    def forward(self, x_t, h_prev, c_prev):
        """
        x_t:[B, V]
        h_prev:[B, H]
        c_prev:[B, H]
        
        return:
            h_t:[B, H]
            c_t:[B, H]
            gates:用于观察四个门
        """
        
        #[B, 4H]
        gate_values = (
            x_t @ self.W_x
            + h_prev @ self.W_h
            + self.b
        )
        
        #将最后一维平均切分成四份
        #每一份都是[B, H]
        i_pre, f_pre, g_pre, o_pre = (
            gate_values.chunk(4, dim=-1) #-1代表从后往前数的第一个维度
        )
        
        input_gate = torch.sigmoid(i_pre)
        forget_gate = torch.sigmoid(f_pre)
        candidate = torch.tanh(g_pre)
        output_gate = torch.sigmoid(o_pre)
        
        # 对应元素相乘使用*
        c_t = (
            forget_gate * c_prev +
            input_gate * candidate
        )
        
        h_t = output_gate * torch.tanh(c_t)
        
        gates = {
            "input": input_gate,
            "forget": forget_gate,
            "candidate": candidate,
            "output": output_gate,
        }
        
        return h_t, c_t, gates
 
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.lstm_cell = LSTMCellFromScratch(
            input_size = vocab_size,
            hidden_size = hidden_size,
        )   
        
        #隐藏状态到字符logits
        self.W_hy = nn.Parameter(
            torch.empty(
                hidden_size,
                vocab_size,
            )
        )
        
        self.b_y = nn.Parameter(
            torch.zeros(vocab_size)
        )
        
        nn.init.xavier_uniform_(self.W_hy)
    
    def forward(self, input_ids, initial_state = None):
        """
        input_ids:
            [B, T]

        initial_state:
            None
            或者 (h_0, c_0)
            两者尺寸均为 [B, H]

        return:
            logits: [B, T, V]
            state:  (h_T, c_T)
        """
        batch_size, seq_len = input_ids.shape
        
        #[B, V] -> [B, T, V]
        x = F.one_hot(
            input_ids,
            num_classes = self.vocab_size,
        ).to(torch.float32)
        
        if initial_state is None:
            h_t = torch.zeros(
                batch_size,
                self.hidden_size,
                device = input_ids.device
            )
            
            c_t = torch.zeros(
                batch_size,
                self.hidden_size,
                device = input_ids.device,
            )
        else:
            h_t, c_t = inital_state
            
        logits_at_each_step = []
        
        for t in range(seq_len):
            #[B, V]
            x_t = x[:, t, :]
            
            h_t, c_t, _ = self.lstm_cell(
                x_t = x_t,
                h_prev = h_t,
                c_prev = c_t,
            )
            
            logits_t = (
                h_t @ self.W_hy
                + self.b_y
            )
            
            logits_at_each_step.append(
                logits_t
            )
        
        logits = torch.stack(
            logits_at_each_step,
            dim = 1,
        )
        
        final_state = (h_t, c_t)
        
        return logits, final_state

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
print("输入：", repr(input_text))
print("目标：", repr(target_text))
print("input_ids.shape：", input_ids.shape)
print("target_ids.shape：", target_ids.shape)

hidden_size = 16

model = CharLSTM(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
).to(device)

number_of_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print()
print("可训练参数量：", number_of_parameters)

for name, parameter in model.named_parameters():
    print(name, parameter.shape)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.03,
)

max_epochs = 1000
max_grad_norm = 0.25

for epoch in range(1, max_epochs + 1):
    model.train()

    logits, final_state = model(input_ids)

    h_last, c_last = final_state

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    grad_norm_before = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm,
        )
    )

    optimizer.step()

    with torch.no_grad():
        predicted_ids = logits.argmax(dim=-1)

        accuracy = (
            predicted_ids == target_ids
        ).float().mean().item()

    if epoch == 1 or epoch % 20 == 0:
        was_clipped = (
            grad_norm_before.item()
            > max_grad_norm
        )

        print(
            f"epoch={epoch:4d}, "
            f"loss={loss.item():.6f}, "
            f"accuracy={accuracy:.2%}, "
            f"grad_before="
            f"{grad_norm_before.item():.6f}, "
            f"clipped={was_clipped}"
        )

    if accuracy == 1.0 and loss.item() < 0.02:
        print(
            f"epoch={epoch:4d}, "
            f"loss={loss.item():.6f}, "
            f"accuracy={accuracy:.2%}"
        )
        print("达到过拟合目标，提前停止。")
        break

model.eval()

with torch.no_grad():
    logits, final_state = model(input_ids)
    predicted_ids = logits.argmax(dim=-1)

h_last, c_last = final_state

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
print("h_last.shape：", h_last.shape)
print("c_last.shape：", c_last.shape)

assert logits.shape == (
    1,
    len(input_text),
    vocab_size,
)

assert h_last.shape == (
    1,
    hidden_size,
)

assert c_last.shape == (
    1,
    hidden_size,
)

assert torch.isfinite(logits).all()
assert torch.isfinite(h_last).all()
assert torch.isfinite(c_last).all()

print("CharLSTM 训练闭环检查通过。")