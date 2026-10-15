import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size =  hidden_size
        
        #参数尺寸：
        # W_xh:[V, H]
        # W_hh:[H, H]
        # b_h: [H]
        
        self.W_xh = nn.Parameter(
            torch.empty(input_size, hidden_size)    
        )
        
        self.W_hh = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        
        self.b_h = nn.Parameter(
            torch.zeros(hidden_size)
        )
        
        #tanh网络适合使用Xavier初始化
        nn.init.xavier_uniform_(self.W_xh)
        nn.init.xavier_uniform_(self.W_hh)
    
    def forward(self, x_t, h_prev):
        """
        x_t:    [B, V]
        h_prev: [B, H]

        return:
            h_t: [B, H]
        """
        input_part = x_t @ self.W_xh
        hidden_part = h_prev @ self.W_hh
        
        h_t = torch.tanh(input_part + hidden_part + self.b_h)
        
        return h_t
    
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
hidden_size = 8
batch_size = 1

print("字符表：", chars)
print("字符到索引：", char_to_idx)
print("vocab_size =", vocab_size)


# --------------------------------------------------
# 2. 取第一个时间步的字符 h
# --------------------------------------------------

current_char = text[0]
current_id = torch.tensor(
    [char_to_idx[current_char]],
    dtype=torch.long,
)

print()
print("当前字符：", repr(current_char))
print("current_id：", current_id)
print("current_id.shape：", current_id.shape)


# --------------------------------------------------
# 3. 把字符索引转换成 one-hot 向量
# --------------------------------------------------

x_t = F.one_hot(
    current_id,
    num_classes=vocab_size,
).to(torch.float32)

print()
print("x_t：", x_t)
print("x_t.shape：", x_t.shape)


# --------------------------------------------------
# 4. 创建初始隐藏状态 h_0
# --------------------------------------------------

h_prev = torch.zeros(batch_size, hidden_size)

print()
print("h_prev：", h_prev)
print("h_prev.shape：", h_prev.shape)


# --------------------------------------------------
# 5. 执行一个时间步
# --------------------------------------------------

cell = VanillaRNNCell(
    input_size=vocab_size,
    hidden_size=hidden_size,
)

h_t = cell(x_t, h_prev)

print()
print("W_xh.shape：", cell.W_xh.shape)
print("W_hh.shape：", cell.W_hh.shape)
print("b_h.shape：", cell.b_h.shape)
print("h_t：", h_t)
print("h_t.shape：", h_t.shape)


# --------------------------------------------------
# 6. 基本验证
# --------------------------------------------------

assert x_t.shape == (batch_size, vocab_size)
assert h_prev.shape == (batch_size, hidden_size)
assert h_t.shape == (batch_size, hidden_size)

# tanh 的输出范围应在 [-1, 1]
assert torch.all(h_t >= -1.0)
assert torch.all(h_t <= 1.0)

print()
print("所有检查通过。")
        
        