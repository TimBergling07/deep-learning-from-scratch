import torch

from couplet_data import CoupletTokenizer
from tinyGPT import TinyGPT


VOCAB_PATH = "couplet_data/vocab.json"
CHECKPOINT_PATH = "best_couplet_gpt.pt"

TEMPERATURE = 0.7
TOP_K = 20


@torch.inference_mode()
def generate_couplet(
    model,
    tokenizer,
    upper_line,
    device,
):
    upper_ids = tokenizer.encode_characters(
        upper_line
    )

    prompt = (
        [tokenizer.bos_id]
        + upper_ids
        + [tokenizer.sep_id]
    )

    token_ids = torch.tensor(
        [prompt],
        dtype=torch.long,
        device=device,
    )

    # 一次处理上联，建立 KV Cache。
    logits, _, cache = model(
        token_ids,
        use_cache=True,
    )

    generated_ids = []

    # 强制生成与上联相同的字数。
    for position in range(len(upper_ids)):
        next_logits = logits[:, -1, :]
        next_logits = next_logits / TEMPERATURE

        # 禁止在正文中生成特殊 token。
        forbidden_ids = [
            tokenizer.pad_id,
            tokenizer.unk_id,
            tokenizer.bos_id,
            tokenizer.sep_id,
            tokenizer.eos_id,
        ]

        next_logits[:, forbidden_ids] = float("-inf")

        k = min(
            TOP_K,
            next_logits.shape[-1],
        )

        top_values, _ = torch.topk(
            next_logits,
            k=k,
            dim=-1,
        )

        threshold = top_values[:, -1:]

        next_logits = next_logits.masked_fill(
            next_logits < threshold,
            float("-inf"),
        )

        probabilities = torch.softmax(
            next_logits,
            dim=-1,
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        generated_ids.append(next_token.item())

        if position + 1 < len(upper_ids):
            logits, _, cache = model(
                next_token,
                past_key_values=cache,
                use_cache=True,
            )

    return tokenizer.decode(generated_ids)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("正在加载模型……")

    tokenizer = CoupletTokenizer.load(
        VOCAB_PATH
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    config = checkpoint["config"]

    model = TinyGPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
    ).to(device)

    # 恢复训练时的权重共享结构。
    model.lm_head.weight = (
        model.token_embedding.weight
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    print(
        f"模型加载完成：step={checkpoint['step']}, "
        f"val_loss={checkpoint['val_loss']:.4f}"
    )

    print()
    print("输入上联即可生成下联。")
    print("输入 q 退出；重复输入可生成不同答案。")
    print()

    while True:
        try:
            upper_line = input("上联：").strip()

            if upper_line.lower() in {
                "q",
                "quit",
                "exit",
            }:
                print("再见！")
                break

            if not upper_line:
                continue

            if len(upper_line) > 32:
                print("当前模型只支持最多 32 个字符。\n")
                continue

            unknown_characters = [
                char
                for char in upper_line
                if char not in tokenizer.token_to_id
            ]

            if unknown_characters:
                print(
                    "提示：以下字符不在词表中：",
                    "".join(sorted(set(unknown_characters))),
                )

            lower_line = generate_couplet(
                model,
                tokenizer,
                upper_line,
                device,
            )

            print("下联：" + lower_line)
            print()

        except KeyboardInterrupt:
            print("\n再见！")
            break


if __name__ == "__main__":
    main()