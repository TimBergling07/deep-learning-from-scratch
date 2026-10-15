from pathlib import Path

import torch
import re
import readline

from couplet_data import CoupletTokenizer
from coupletTransformer import CoupletTransformer


BASE_DIR = Path(__file__).resolve().parent

VOCAB_PATH = BASE_DIR / "couplet_data" / "vocab.json"
CHECKPOINT_PATH = BASE_DIR / "best_couplet_seq2seq.pt"

TEMPERATURE = 0.8
TOP_K = 30
NUM_CANDIDATES = 8

readline.parse_and_bind("set editing-mode emacs")

# Delete：删除光标右侧字符。
readline.parse_and_bind(r'"\e[3~": delete-char')

# Backspace：删除光标左侧字符。
readline.parse_and_bind(r'"\C-?": backward-delete-char')
readline.parse_and_bind(r'"\C-h": backward-delete-char')


def clean_upper_line(text):
    # 防止终端控制序列残留在最终输入中。
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)

    # 清除其他 ASCII 控制字符。
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)

    return text.strip()

def load_model(device):
    tokenizer = CoupletTokenizer.load(VOCAB_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]

    model = CoupletTransformer(
        vocab_size=config["vocab_size"],
        pad_id=config["pad_id"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
    )

    # 与训练时保持相同的权重共享关系。
    model.lm_head.weight = model.token_embedding.weight

    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    print("device:", device)
    print(
        "loaded best checkpoint:",
        checkpoint["step"],
        checkpoint["val_loss"],
    )

    return model, tokenizer


@torch.inference_mode()
def encode_upper(model, tokenizer, upper_line, device):
    upper_ids = tokenizer.encode_characters(upper_line)

    if not upper_ids:
        raise ValueError("上联不能为空")

    # 与训练时 BOS + 下联的长度限制保持一致。
    if len(upper_ids) + 1 > model.max_seq_len:
        raise ValueError(
            f"上联最多支持 {model.max_seq_len - 1} 个字符"
        )

    source_ids = torch.tensor(
        [upper_ids],
        dtype=torch.long,
        device=device,
    )

    source_padding_mask = source_ids.eq(tokenizer.pad_id)

    memory = model.encode(
        source_ids,
        source_padding_mask,
    )

    return memory, source_padding_mask, len(upper_ids)


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    memory,
    source_padding_mask,
    output_length,
    count=1,
    greedy=False,
    temperature=0.8,
    top_k=30,
):
    if not greedy:
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k 必须至少为 1")

    memory = memory.expand(count, -1, -1)
    source_padding_mask = source_padding_mask.expand(count, -1)

    step_input = torch.full(
        (count, 1),
        tokenizer.bos_id,
        dtype=torch.long,
        device=memory.device,
    )

    cache = None
    generated_tokens = []

    forbidden_ids = [
        tokenizer.pad_id,
        tokenizer.unk_id,
        tokenizer.bos_id,
        tokenizer.sep_id,
        tokenizer.eos_id,
    ]

    for _ in range(output_length):
        logits, cache = model.decode(
            step_input,
            memory,
            source_padding_mask,
            past_key_values=cache,
            use_cache=True,
        )

        next_logits = logits[:, -1, :].clone()
        next_logits[:, forbidden_ids] = float("-inf")

        if greedy:
            next_token = next_logits.argmax(
                dim=-1,
                keepdim=True,
            )
        else:
            next_logits = next_logits / temperature

            if top_k is not None:
                k = min(top_k, next_logits.size(-1))
                top_values, _ = torch.topk(
                    next_logits, k=k, dim=-1
                )

                next_logits = next_logits.masked_fill(
                    next_logits < top_values[:, -1:],
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

        generated_tokens.append(next_token)

        # 下一轮只处理刚生成的字符。
        step_input = next_token

    generated_ids = torch.cat(
        generated_tokens,
        dim=1,
    ).tolist()

    return [
        tokenizer.decode(token_ids)
        for token_ids in generated_ids
    ]

def reorder_cache(cache, parent_indices):
    reordered = []

    for self_kv, cross_kv in cache:
        self_k, self_v = self_kv
        cross_k, cross_v = cross_kv

        reordered.append((
            (
                self_k.index_select(0, parent_indices),
                self_v.index_select(0, parent_indices),
            ),
            (
                cross_k.index_select(0, parent_indices),
                cross_v.index_select(0, parent_indices),
            ),
        ))

    return reordered

@torch.inference_mode()
def beam_search(
    model,
    tokenizer,
    memory,
    source_padding_mask,
    output_length,
    beam_size=5,
):
    if beam_size < 1:
        raise ValueError("beam_size 必须至少为 1")

    sequences = torch.tensor(
        [[tokenizer.bos_id]],
        dtype=torch.long,
        device=memory.device,
    )

    step_input = sequences
    beam_scores = torch.zeros(1, device=memory.device)
    cache = None

    forbidden_ids = [
        tokenizer.pad_id,
        tokenizer.unk_id,
        tokenizer.bos_id,
        tokenizer.sep_id,
        tokenizer.eos_id,
    ]

    for _ in range(output_length):
        current_beams = sequences.size(0)

        logits, present_cache = model.decode(
            step_input,
            memory.expand(current_beams, -1, -1),
            source_padding_mask.expand(current_beams, -1),
            past_key_values=cache,
            use_cache=True,
        )

        log_probs = torch.log_softmax(
            logits[:, -1, :],
            dim=-1,
        )
        log_probs[:, forbidden_ids] = float("-inf")

        vocab_size = log_probs.size(-1)

        candidate_scores = (
            beam_scores.unsqueeze(1) + log_probs
        )

        valid_extensions = current_beams * (
            vocab_size - len(set(forbidden_ids))
        )

        beam_scores, flat_indices = torch.topk(
            candidate_scores.reshape(-1),
            k=min(beam_size, valid_extensions),
        )

        parent_indices = flat_indices // vocab_size
        next_token_ids = flat_indices % vocab_size

        sequences = torch.cat(
            [
                sequences[parent_indices],
                next_token_ids.unsqueeze(1),
            ],
            dim=1,
        )

        # 缓存必须与新的候选路径对应。
        cache = reorder_cache(
            present_cache,
            parent_indices,
        )

        step_input = next_token_ids.unsqueeze(1)

    generated_ids = sequences[:, 1:].tolist()
    scores = beam_scores.tolist()

    return [
        (tokenizer.decode(token_ids), score)
        for token_ids, score in zip(generated_ids, scores)
    ]

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, tokenizer = load_model(device)

    print()
    print("输入上联，回车生成；输入 q 退出。")
    print(
        f"采样设置：temperature={TEMPERATURE}, "
        f"top_k={TOP_K}, 候选数={NUM_CANDIDATES}"
    )

    while True:
        try:
            upper_line = clean_upper_line(input("\n上联："))
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if upper_line.lower() in {"q", "quit", "exit"}:
            break

        if not upper_line:
            continue

        try:
            memory, padding_mask, length = encode_upper(
                model,
                tokenizer,
                upper_line,
                device,
            )
        except ValueError as error:
            print(error)
            continue

        greedy_result = generate(
            model,
            tokenizer,
            memory,
            padding_mask,
            output_length=length,
            greedy=True,
        )[0]
        
        beam_results = beam_search(
            model,
            tokenizer,
            memory,
            padding_mask,
            output_length=length,
            beam_size=5,
        )
        
        candidates = generate(
            model,
            tokenizer,
            memory,
            padding_mask,
            output_length=length,
            count=NUM_CANDIDATES,
            temperature=TEMPERATURE,
            top_k=TOP_K,
        )

        print("\nGreedy：", greedy_result)

        print("\nBeam Search：")
        for index, (candidate, score) in enumerate(
        beam_results,
            start=1,
        ):
            print(
                f"{index:2d}. {candidate} "
                f"| score {score:.3f}"
            )

        print("\n采样候选：")
        for index, candidate in enumerate(candidates, start=1):
            print(f"{index:2d}. {candidate}")

if __name__ == "__main__":
    main()