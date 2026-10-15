import torch

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

print("原始文本：", repr(text))
print("字符表：", chars)
print("词表大小：", len(chars))
print("字符 -> 索引：", char_to_idx)
print("索引 -> 字符：", idx_to_char)

start = 0
seq_len = 4

input_chars = text[start : start + seq_len]
target_chars = text[start+1 : start + seq_len + 1]

input_ids = torch.tensor(
    [char_to_idx[char] for char in input_chars],
    dtype = torch.long,
)

target_ids = torch.tensor(
    [char_to_idx[char] for char in target_chars],
    dtype = torch.long,
)

print()
print("输入字符：", repr(input_chars))
print("目标字符：", repr(target_chars))
print("输入索引：", input_ids)
print("目标索引：", target_ids)
print("输入尺寸：", input_ids.shape)
print("目标尺寸：", target_ids.shape)

decoded_input = "".join(idx_to_char[index.item()] for index in input_ids)
decoded_target = "".join(idx_to_char[index.item()] for index in target_ids)

print()
print("还原输入：", repr(decoded_input))
print("还原目标：", repr(decoded_target))

assert decoded_input == input_chars
assert decoded_target == target_chars