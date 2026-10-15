import json

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    id_to_token = vocab["id_to_token"]
    token_to_id = {
        token: token_id
        for token_id, token in enumerate(id_to_token)
    }

    return id_to_token, token_to_id


class CoupletDataset(Dataset):
    def __init__(
        self,
        path,
        token_to_id,
        max_seq_len=72,
    ):
        self.token_to_id = token_to_id
        self.unk_id = token_to_id["<UNK>"]
        self.bos_id = token_to_id["<BOS>"]
        self.eos_id = token_to_id["<EOS>"]

        self.pairs = []

        with open(path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                sample = json.loads(line)

                # 兼容两种常见的上、下联字段名。
                if "up" in sample and "down" in sample:
                    upper = sample["up"]
                    lower = sample["down"]
                elif "upper" in sample and "lower" in sample:
                    upper = sample["upper"]
                    lower = sample["lower"]
                else:
                    raise ValueError(
                        f"{path} 第 {line_number} 行字段无法识别："
                        f"{list(sample.keys())}"
                    )

                if not upper or not lower:
                    raise ValueError(
                        f"{path} 第 {line_number} 行存在空上联或下联"
                    )

                # 下联输入额外包含一个 BOS。
                if max(len(upper), len(lower) + 1) > max_seq_len:
                    raise ValueError(
                        f"{path} 第 {line_number} 行超过 "
                        f"max_seq_len={max_seq_len}"
                    )

                self.pairs.append((upper, lower))

    def __len__(self):
        return len(self.pairs)

    def encode(self, text):
        return [
            self.token_to_id.get(char, self.unk_id)
            for char in text
        ]

    def __getitem__(self, index):
        upper, lower = self.pairs[index]

        source_ids = self.encode(upper)
        lower_ids = self.encode(lower)

        decoder_input = [self.bos_id] + lower_ids
        labels = lower_ids + [self.eos_id]

        return (
            torch.tensor(source_ids, dtype=torch.long),
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


class CoupletCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, samples):
        sources, decoder_inputs, labels = zip(*samples)

        # 三组序列分别进行右侧补齐。
        sources = pad_sequence(
            sources,
            batch_first=True,
            padding_value=self.pad_id,
        )

        decoder_inputs = pad_sequence(
            decoder_inputs,
            batch_first=True,
            padding_value=self.pad_id,
        )

        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.pad_id,
        )

        return sources, decoder_inputs, labels