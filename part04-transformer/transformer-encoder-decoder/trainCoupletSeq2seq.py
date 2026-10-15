import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 继续复用原来的 tokenizer，生成时需要它。
from couplet_data import CoupletTokenizer

from seq2seq_data import (
    load_vocab,
    CoupletDataset,
    CoupletCollator,
)
from coupletTransformer import CoupletTransformer


# ---------- 配置 ----------

TRAIN_PATH = "couplet_data/train.jsonl"
VAL_PATH = "couplet_data/val.jsonl"
VOCAB_PATH = "couplet_data/vocab.json"
CHECKPOINT_PATH = "best_couplet_seq2seq.pt"

NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

MAX_SEQ_LEN = 72

D_MODEL = 384
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.15

BATCH_SIZE = 64
VAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1

MAX_STEPS = 50_000
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

    for source_ids, decoder_input, labels in data_loader:
        source_ids = source_ids.to(
            device, non_blocking=True
        )
        decoder_input = decoder_input.to(
            device, non_blocking=True
        )
        labels = labels.to(
            device, non_blocking=True
        )

        logits = model(source_ids, decoder_input)

        # 累计有效 token 的 loss 总和。
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=model.pad_id,
            reduction="sum",
        )

        total_loss += loss_sum.item()
        total_tokens += (
            labels != model.pad_id
        ).sum().item()

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

    if not upper_ids:
        raise ValueError("上联不能为空")

    # Encoder 只接收上联，不拼接 BOS、SEP。
    source_ids = torch.tensor(
        [upper_ids],
        dtype=torch.long,
        device=device,
    )

    source_padding_mask = source_ids.eq(
        tokenizer.pad_id
    )

    # 完整上联只编码一次。
    memory = model.encode(
        source_ids,
        source_padding_mask,
    )

    # Decoder 从 BOS 开始生成。
    decoder_input = torch.tensor(
        [[tokenizer.bos_id]],
        dtype=torch.long,
        device=device,
    )

    generated_ids = []

    forbidden_ids = [
        tokenizer.pad_id,
        tokenizer.unk_id,
        tokenizer.bos_id,
        tokenizer.sep_id,
        tokenizer.eos_id,
    ]

    # 沿用原来的等长生成规则。
    for _ in range(len(upper_ids)):
        logits = model.decode(
            decoder_input,
            memory,
            source_padding_mask,
        )

        next_logits = (
            logits[:, -1, :] / temperature
        )

        next_logits[:, forbidden_ids] = float("-inf")

        if top_k is not None:
            k = min(top_k, next_logits.size(-1))

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

        decoder_input = torch.cat(
            [decoder_input, next_token],
            dim=1,
        )

    return tokenizer.decode(generated_ids)


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = CoupletTokenizer.load(VOCAB_PATH)

    _, token_to_id = load_vocab(VOCAB_PATH)

    train_dataset = CoupletDataset(
        TRAIN_PATH,
        token_to_id,
        max_seq_len=MAX_SEQ_LEN,
    )

    val_dataset = CoupletDataset(
        VAL_PATH,
        token_to_id,
        max_seq_len=MAX_SEQ_LEN,
    )

    collate_fn = CoupletCollator(tokenizer.pad_id)

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

    model = CoupletTransformer(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_id,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    ).to(device)

    model.apply(initialize_weights)

    # 沿用原模型的输入输出权重共享。
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

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for micro_step in range(
            GRADIENT_ACCUMULATION_STEPS
        ):
            try:
                source_ids, decoder_input, labels = next(
                    train_iterator
                )
            except StopIteration:
                train_iterator = iter(train_loader)
                source_ids, decoder_input, labels = next(
                    train_iterator
                )

            source_ids = source_ids.to(
                device, non_blocking=True
            )
            decoder_input = decoder_input.to(
                device, non_blocking=True
            )
            labels = labels.to(
                device, non_blocking=True
            )

            logits = model(source_ids, decoder_input)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=tokenizer.pad_id,
            )

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
                            "pad_id": tokenizer.pad_id,
                            "d_model": D_MODEL,
                            "num_heads": NUM_HEADS,
                            "num_encoder_layers": NUM_ENCODER_LAYERS,
                            "num_decoder_layers": NUM_DECODER_LAYERS,
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