"""HTTP bridge between the web UI and the locally trained Couplet GPT.

Put this file beside tinyGPT.py, best_couplet_gpt.pt and couplet_data/, then run:
    pip install flask flask-cors
    python model_server.py
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from flask_cors import CORS

from tinyGPT import TinyGPT


BASE_DIR = Path(__file__).resolve().parent
VOCAB_PATH = BASE_DIR / "couplet_data" / "vocab.json"
CHECKPOINT_PATH = BASE_DIR / "best_couplet_gpt.pt"
HOST = "127.0.0.1"
PORT = 7861


class CoupletTokenizer:
    def __init__(self, vocab_path: Path):
        with vocab_path.open("r", encoding="utf-8") as file:
            vocab = json.load(file)

        self.itos = vocab["id_to_token"]
        self.stoi = {
            token: token_id
            for token_id, token in enumerate(self.itos)
        }
        self.pad_id = self.stoi["<PAD>"]
        self.unk_id = self.stoi["<UNK>"]
        self.bos_id = self.stoi["<BOS>"]
        self.sep_id = self.stoi["<SEP>"]
        self.eos_id = self.stoi["<EOS>"]

    def encode_upper(self, text: str) -> list[int]:
        return [self.bos_id, *(self.stoi.get(char, self.unk_id) for char in text), self.sep_id]

    def decode(self, token_ids: list[int]) -> str:
        special = {self.pad_id, self.unk_id, self.bos_id, self.sep_id, self.eos_id}
        return "".join(self.itos[token_id] for token_id in token_ids if token_id not in special)

    def unknown_characters(self, text: str) -> list[str]:
        return list(dict.fromkeys(char for char in text if char not in self.stoi))


def load_model():
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"找不到词表：{VOCAB_PATH}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{CHECKPOINT_PATH}")

    tokenizer = CoupletTokenizer(VOCAB_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyGPT(
        vocab_size=len(tokenizer.itos),
        max_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    model.lm_head.weight = model.token_embedding.weight
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return tokenizer, model, device


TOKENIZER, MODEL, DEVICE = load_model()
MODEL_LOCK = threading.Lock()
FORBIDDEN_IDS = [TOKENIZER.pad_id, TOKENIZER.unk_id, TOKENIZER.bos_id, TOKENIZER.sep_id, TOKENIZER.eos_id]


@torch.inference_mode()
def generate_batch(upper: str, temperature: float, top_k: int, count: int) -> list[str]:
    # Generate extra rows so duplicate candidates can be removed cheaply.
    batch_size = min(max(count * 2, count), 16)
    prompt = torch.tensor(
        [TOKENIZER.encode_upper(upper)], dtype=torch.long, device=DEVICE
    ).repeat(batch_size, 1)

    logits, cache = unpack_cached_output(MODEL(prompt, use_cache=True))
    generated = [[] for _ in range(batch_size)]

    for _ in range(len(upper)):
        next_logits = logits[:, -1, :] / temperature
        next_logits[:, FORBIDDEN_IDS] = float("-inf")

        actual_k = min(top_k, next_logits.size(-1))
        top_values, _ = torch.topk(next_logits, actual_k, dim=-1)
        cutoff = top_values[:, -1].unsqueeze(-1)
        next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))

        probabilities = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        for row, token_id in enumerate(next_token.squeeze(-1).tolist()):
            generated[row].append(token_id)

        logits, cache = unpack_cached_output(
            MODEL(next_token, past_key_values=cache, use_cache=True)
        )

    unique: list[str] = []
    for row in generated:
        candidate = TOKENIZER.decode(row)
        if candidate and candidate not in unique:
            unique.append(candidate)
        if len(unique) == count:
            break
    return unique


def unpack_cached_output(output):
    """Accept both (logits, cache) and (logits, loss, cache) forward APIs."""
    if not isinstance(output, tuple):
        raise TypeError("TinyGPT.forward(use_cache=True) 必须返回 tuple")
    if len(output) == 2:
        return output[0], output[1]
    if len(output) == 3:
        return output[0], output[2]
    raise ValueError(f"无法识别 TinyGPT.forward 的 {len(output)} 个返回值")


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "device": str(DEVICE), "vocabularySize": len(TOKENIZER.itos)})


@app.post("/api/generate")
def generate():
    payload = request.get_json(silent=True) or {}
    upper = str(payload.get("upper", "")).strip()
    if not upper:
        return jsonify({"error": "请先输入上联"}), 400
    if len(upper) > 32:
        return jsonify({"error": "上联最长 32 个字符"}), 400

    try:
        temperature = min(max(float(payload.get("temperature", 0.8)), 0.4), 1.2)
        top_k = min(max(int(payload.get("topK", 30)), 1), len(TOKENIZER.itos))
        count = min(max(int(payload.get("count", 4)), 1), 8)
    except (TypeError, ValueError):
        return jsonify({"error": "采样参数格式不正确"}), 400

    started = time.perf_counter()
    with MODEL_LOCK:
        candidates = generate_batch(upper, temperature, top_k, count)
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    return jsonify({
        "candidates": candidates,
        "elapsedMs": elapsed_ms,
        "device": str(DEVICE),
        "unknownCharacters": TOKENIZER.unknown_characters(upper),
    })


if __name__ == "__main__":
    print(f"模型设备: {DEVICE}")
    print(f"网页 API: http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
