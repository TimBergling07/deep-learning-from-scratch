import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

from datasets import load_dataset


OUTPUT_DIR = Path("couplet_data")

MIN_LENGTH = 4
MAX_LENGTH = 32

PUNCTUATION = set(
    "，。！？；：、,.!?;:"
)


def normalize_text(text):
    """
    NFC：统一部分 Unicode 表示，但不把中文标点强行转成英文。
    """
    text = unicodedata.normalize("NFC", text)

    # 对联中通常不需要空格、制表符和换行。
    text = "".join(text.split())

    return text


def punctuation_positions(text):
    """
    只比较标点所在位置，不要求两边使用相同标点。
    """
    return tuple(
        index
        for index, char in enumerate(text)
        if char in PUNCTUATION
    )


def clean_pair(up, down):
    if not isinstance(up, str) or not isinstance(down, str):
        return None, "not_string"

    up = normalize_text(up)
    down = normalize_text(down)

    if not up or not down:
        return None, "empty"

    # 正规对联应当上下联等长。
    if len(up) != len(down):
        return None, "unequal_length"

    if not MIN_LENGTH <= len(up) <= MAX_LENGTH:
        return None, "length"

    if up == down:
        return None, "identical"

    # 去掉包含英文和阿拉伯数字的样本，
    # 减少网址、编号和异常数据。
    if any(
        char.isascii() and char.isalnum()
        for char in up + down
    ):
        return None, "ascii_alnum"

    # 逗号等标点的位置应当大致对应。
    if punctuation_positions(up) != punctuation_positions(down):
        return None, "punctuation"

    return {
        "up": up,
        "down": down,
    }, None


def choose_split(up):
    """
    根据上联内容确定 split。

    相同上联一定进入同一个 split，避免同一道题同时出现在
    训练集和验证集。
    """
    digest = hashlib.sha256(
        up.encode("utf-8")
    ).digest()

    bucket = int.from_bytes(
        digest[:4],
        byteorder="big",
    ) % 100

    if bucket < 98:
        return "train"
    if bucket == 98:
        return "val"
    return "test"


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("正在下载并读取对联数据……")

    dataset = load_dataset(
        "wb14123/couplet",
        split="train",
    )

    print("原始样本数:", len(dataset))
    print("字段:", dataset.column_names)
    print("第一条原始样本:", dataset[0])

    output_paths = {
        "train": OUTPUT_DIR / "train.jsonl",
        "val": OUTPUT_DIR / "val.jsonl",
        "test": OUTPUT_DIR / "test.jsonl",
    }

    output_files = {
        split: path.open(
            "w",
            encoding="utf-8",
        )
        for split, path in output_paths.items()
    }

    split_counts = Counter()
    rejected_counts = Counter()
    train_char_counts = Counter()

    length_sum = Counter()
    minimum_length = {
        "train": MAX_LENGTH + 1,
        "val": MAX_LENGTH + 1,
        "test": MAX_LENGTH + 1,
    }
    maximum_length = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    # 防止完全相同的上下联组合重复出现。
    seen_pairs = set()

    examples = []

    try:
        for index, row in enumerate(dataset):
            pair, reason = clean_pair(
                row["up"],
                row["down"],
            )

            if pair is None:
                rejected_counts[reason] += 1
                continue

            pair_key = (
                pair["up"],
                pair["down"],
            )

            if pair_key in seen_pairs:
                rejected_counts["duplicate"] += 1
                continue

            seen_pairs.add(pair_key)

            split = choose_split(pair["up"])
            pair_length = len(pair["up"])

            json.dump(
                pair,
                output_files[split],
                ensure_ascii=False,
            )
            output_files[split].write("\n")

            split_counts[split] += 1
            length_sum[split] += pair_length

            minimum_length[split] = min(
                minimum_length[split],
                pair_length,
            )

            maximum_length[split] = max(
                maximum_length[split],
                pair_length,
            )

            # 词表只能根据训练集构建，避免验证集信息泄漏。
            if split == "train":
                train_char_counts.update(pair["up"])
                train_char_counts.update(pair["down"])

            if len(examples) < 5:
                examples.append(pair)

            if (index + 1) % 100_000 == 0:
                print(
                    f"已处理 {index + 1:,} / "
                    f"{len(dataset):,}"
                )

    finally:
        for file in output_files.values():
            file.close()

    print("\n清洗完成")

    for split in ("train", "val", "test"):
        count = split_counts[split]

        average_length = (
            length_sum[split] / count
            if count > 0
            else 0.0
        )

        print(
            f"{split:5s}: "
            f"{count:,} 条 | "
            f"平均长度 {average_length:.2f} | "
            f"范围 {minimum_length[split]}"
            f"-{maximum_length[split]}"
        )

    print("\n被过滤的数据:")
    for reason, count in rejected_counts.most_common():
        print(f"{reason:16s}: {count:,}")

    print(
        "\n训练集不同字符数:",
        len(train_char_counts),
    )

    print("\n训练集最高频字符:")
    for char, count in train_char_counts.most_common(30):
        print(repr(char), count)

    print("\n清洗后示例:")
    for pair in examples:
        print("上联：", pair["up"])
        print("下联：", pair["down"])
        print()

    print("输出文件:")
    for split, path in output_paths.items():
        print(
            f"{split:5s}: {path} "
            f"({split_counts[split]:,} 条)"
        )


if __name__ == "__main__":
    main()