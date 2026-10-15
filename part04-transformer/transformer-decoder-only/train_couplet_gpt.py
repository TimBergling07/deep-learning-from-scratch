import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from couplet_data import (
    IGNORE_INDEX,
    CoupletDataset,
    CoupletTokenizer,
    make_collate_fn,
)
from tinyGPT import TinyGPT


# ---------- 配置 ----------

TRAIN_PATH = "couplet_data/train.jsonl"
VAL_PATH = "couplet_data/val.jsonl"
VOCAB_PATH = "couplet_data/vocab.json"
CHECKPOINT_PATH = "best_couplet_gpt.pt"

MAX_SEQ_LEN = 72

D_MODEL = 384
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.15

BATCH_SIZE = 32
VAL_BATCH_SIZE = 64

GRADIENT_ACCUMULATION_STEPS = 2

MAX_STEPS = 30_000
WARMUP_STEPS = 1_000

MAX_LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5

LOG_INTERVAL = 100
EVAL_INTERVAL = 1_000
EARLY_STOPPING_PATIENCE = 6


def get_learning_rate(step):
    if step < WARMUP_STEPS:
        return (
            MAX_LEARNING_RATE
            * (step + 1)
            / WARMUP_STEPS
        )

    progress = (
        (step - WARMUP_STEPS)
        / (MAX_STEPS - WARMUP_STEPS)
    )

    progress = min(max(progress, 0.0), 1.0)

    cosine = 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )

    return (
        MIN_LEARNING_RATE
        + cosine
        * (MAX_LEARNING_RATE - MIN_LEARNING_RATE)
    )


def initialize_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(
            module.weight,
            mean=0.0,
            std=0.02,
        )

        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(
            module.weight,
            mean=0.0,
            std=0.02,
        )


@torch.inference_mode()
def evaluate(model, data_loader, device):
    previous_mode = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for input_ids, targets in data_loader:
        input_ids = input_ids.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        _, loss = model(input_ids, targets)

        supervised_tokens = (
            targets != IGNORE_INDEX
        ).sum().item()

        total_loss += loss.item() * supervised_tokens
        total_tokens += supervised_tokens

    model.train(previous_mode)

    return total_loss / total_tokens


@torch.inference_mode()
def generate_couplet(
    model,
    tokenizer,
    upper_line,
    device,
    temperature=0.8,
    top_k=30,
):
    model.eval()

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

    # Prefill 并建立 KV Cache。
    logits, _, cache = model(
        token_ids,
        use_cache=True,
    )

    generated_ids = []

    # 对联上下联等长，生成固定数量的汉字。
    for position in range(len(upper_ids)):
        next_logits = logits[:, -1, :]
        next_logits = next_logits / temperature

        # 这些特殊 token 不应该出现在下联正文。
        forbidden_ids = [
            tokenizer.pad_id,
            tokenizer.unk_id,
            tokenizer.bos_id,
            tokenizer.sep_id,
            tokenizer.eos_id,
        ]

        next_logits[:, forbidden_ids] = float("-inf")

        if top_k is not None:
            k = min(top_k, next_logits.shape[-1])

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
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = CoupletTokenizer.load(VOCAB_PATH)

    train_dataset = CoupletDataset(
        TRAIN_PATH,
        tokenizer,
    )

    val_dataset = CoupletDataset(
        VAL_PATH,
        tokenizer,
    )

    collate_fn = make_collate_fn(
        tokenizer.pad_id
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    model = TinyGPT(
        vocab_size=len(tokenizer),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    ).to(device)

    model.apply(initialize_weights)

    # 共享输入和输出的字符表示。
    model.lm_head.weight = model.token_embedding.weight

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("device:", device)
    print("vocabulary size:", len(tokenizer))
    print("train examples:", len(train_dataset))
    print("val examples:", len(val_dataset))
    print("parameters:", parameter_count)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MAX_LEARNING_RATE,
        weight_decay=0.1,
    )

    best_val_loss = float("inf")
    evaluations_without_improvement = 0
    running_loss = 0.0

    model.train()

    train_iterator = iter(train_loader)
    
    for step in range(1, MAX_STEPS + 1):
        learning_rate = get_learning_rate(step - 1)

        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        try:
            input_ids, targets = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            input_ids, targets = next(train_iterator)

        input_ids = input_ids.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        _, loss = model(input_ids, targets)

        optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0

        for micro_step in range(
            GRADIENT_ACCUMULATION_STEPS
        ):
            try:
                input_ids, targets = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                input_ids, targets = next(train_iterator)

            input_ids = input_ids.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )

            _, loss = model(input_ids, targets)

            # 除以累计次数，保持梯度尺度与 batch=64 接近。
            scaled_loss = (
                loss / GRADIENT_ACCUMULATION_STEPS
            )

            scaled_loss.backward()

            accumulated_loss += (
                loss.item()
                / GRADIENT_ACCUMULATION_STEPS
            )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        running_loss += accumulated_loss

        if step % LOG_INTERVAL == 0:
            average_train_loss = (
                running_loss / LOG_INTERVAL
            )

            print(
                f"step {step:5d} | "
                f"train {average_train_loss:.4f} | "
                f"lr {learning_rate:.2e}"
            )

            running_loss = 0.0

        if step % EVAL_INTERVAL == 0:
            val_loss = evaluate(
                model,
                val_loader,
                device,
            )

            print(
                f"           val {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                evaluations_without_improvement = 0

                torch.save(
                    {
                        "step": step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "config": {
                            "vocab_size": len(tokenizer),
                            "d_model": D_MODEL,
                            "num_heads": NUM_HEADS,
                            "num_layers": NUM_LAYERS,
                            "max_seq_len": MAX_SEQ_LEN,
                            "dropout": DROPOUT,
                        },
                    },
                    CHECKPOINT_PATH,
                )

                print("           saved best checkpoint")

            else:
                evaluations_without_improvement += 1

                print(
                    "           no improvement:",
                    f"{evaluations_without_improvement}/"
                    f"{EARLY_STOPPING_PATIENCE}",
                )

                if (
                    evaluations_without_improvement
                    >= EARLY_STOPPING_PATIENCE
                ):
                    print("early stopping")
                    break

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    print(
        "\nloaded best checkpoint:",
        checkpoint["step"],
        checkpoint["val_loss"],
    )

    test_upper_lines = [
        "春风又绿江南岸",
        "人工智能开新境",
        "山高水远情难尽",
        "书山有路勤为径",
    ]

    for upper_line in test_upper_lines:
        lower_line = generate_couplet(
            model,
            tokenizer,
            upper_line,
            device,
            temperature=0.8,
            top_k=30,
        )

        print()
        print("上联：", upper_line)
        print("下联：", lower_line)


if __name__ == "__main__":
    main()