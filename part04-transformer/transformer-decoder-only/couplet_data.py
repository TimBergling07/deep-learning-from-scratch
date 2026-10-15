import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
SEP_TOKEN = "<SEP>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    SEP_TOKEN,
    EOS_TOKEN,
]

IGNORE_INDEX = -100


class CoupletTokenizer:
    def __init__(self, id_to_token):
        self.id_to_token = id_to_token

        self.token_to_id = {
            token: index
            for index, token in enumerate(id_to_token)
        }

        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]
        self.bos_id = self.token_to_id[BOS_TOKEN]
        self.sep_id = self.token_to_id[SEP_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]

    def __len__(self):
        return len(self.id_to_token)

    def encode_characters(self, text):
        return [
            self.token_to_id.get(char, self.unk_id)
            for char in text
        ]

    def decode(self, token_ids):
        characters = []

        for token_id in token_ids:
            token = self.id_to_token[token_id]

            if token == EOS_TOKEN:
                break

            if token in SPECIAL_TOKENS:
                continue

            characters.append(token)

        return "".join(characters)

    def save(self, path):
        path = Path(path)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "id_to_token": self.id_to_token,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path):
        with Path(path).open(
            "r",
            encoding="utf-8",
        ) as file:
            content = json.load(file)

        return cls(content["id_to_token"])

    @classmethod
    def build(
        cls,
        train_path,
        min_frequency=2,
    ):
        character_counts = Counter()

        with Path(train_path).open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                pair = json.loads(line)

                character_counts.update(pair["up"])
                character_counts.update(pair["down"])

        # 先按频率降序，再按字符排序，保证结果可复现。
        characters = [
            char
            for char, frequency in sorted(
                character_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if frequency >= min_frequency
        ]

        id_to_token = SPECIAL_TOKENS + characters

        removed_characters = sum(
            1
            for frequency in character_counts.values()
            if frequency < min_frequency
        )

        removed_occurrences = sum(
            frequency
            for frequency in character_counts.values()
            if frequency < min_frequency
        )

        print("原始字符种类:", len(character_counts))
        print("词表保留字符:", len(characters))
        print("特殊 token:", len(SPECIAL_TOKENS))
        print("最终词表大小:", len(id_to_token))
        print("被映射为 UNK 的低频字符种类:", removed_characters)
        print("低频字符总出现次数:", removed_occurrences)

        return cls(id_to_token)


class CoupletDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer):
        self.tokenizer = tokenizer
        self.examples = []

        with Path(jsonl_path).open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                pair = json.loads(line)
                self.examples.append(pair)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        pair = self.examples[index]

        up_ids = self.tokenizer.encode_characters(
            pair["up"]
        )

        down_ids = self.tokenizer.encode_characters(
            pair["down"]
        )

        complete_sequence = (
            [self.tokenizer.bos_id]
            + up_ids
            + [self.tokenizer.sep_id]
            + down_ids
            + [self.tokenizer.eos_id]
        )

        # 标准 next-token 输入和目标。
        input_ids = torch.tensor(
            complete_sequence[:-1],
            dtype=torch.long,
        )

        targets = torch.tensor(
            complete_sequence[1:],
            dtype=torch.long,
        )

        # 忽略：
        # BOS -> 上联字符
        # 上联字符 -> 后续上联字符/SEP
        #
        # 从 SEP -> 下联第一个字符开始计算 loss。
        targets[:len(up_ids) + 1] = IGNORE_INDEX

        return input_ids, targets


def make_collate_fn(pad_id):
    def collate_batch(batch):
        maximum_length = max(
            input_ids.shape[0]
            for input_ids, _ in batch
        )

        batch_size = len(batch)

        input_batch = torch.full(
            size=(batch_size, maximum_length),
            fill_value=pad_id,
            dtype=torch.long,
        )

        target_batch = torch.full(
            size=(batch_size, maximum_length),
            fill_value=IGNORE_INDEX,
            dtype=torch.long,
        )

        for row, (input_ids, targets) in enumerate(batch):
            length = input_ids.shape[0]

            input_batch[row, :length] = input_ids
            target_batch[row, :length] = targets

        return input_batch, target_batch

    return collate_batch


def main():
    train_path = "couplet_data/train.jsonl"
    vocab_path = "couplet_data/vocab.json"

    tokenizer = CoupletTokenizer.build(
        train_path,
        min_frequency=2,
    )

    tokenizer.save(vocab_path)

    train_dataset = CoupletDataset(
        train_path,
        tokenizer,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=make_collate_fn(
            tokenizer.pad_id
        ),
    )

    input_ids, targets = next(iter(loader))

    supervised_positions = (
        targets != IGNORE_INDEX
    ).sum().item()

    print("\n数据集大小:", len(train_dataset))
    print("input batch:", input_ids.shape)
    print("target batch:", targets.shape)
    print("参与 loss 的位置数:", supervised_positions)

    print("\n第一条 input token:")
    print([
        tokenizer.id_to_token[token_id]
        for token_id in input_ids[0].tolist()
    ])

    print("\n第一条 target token:")
    print([
        "忽略"
        if token_id == IGNORE_INDEX
        else tokenizer.id_to_token[token_id]
        for token_id in targets[0].tolist()
    ])

    print("\n词表已保存到:", vocab_path)


if __name__ == "__main__":
    main()