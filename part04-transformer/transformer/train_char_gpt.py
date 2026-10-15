from pathlib import Path

import math
import torch

from tinyGPT import TinyGPT

#configuration

BLOCK_SIZE = 256
BATCH_SIZE = 32

D_MODEL = 384
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.2

MAX_LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WARMUP_STEPS = 300
MAX_STEPS = 7000

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

    cosine = 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )

    return (
        MIN_LEARNING_RATE
        + cosine
        * (MAX_LEARNING_RATE - MIN_LEARNING_RATE)
    )
    
EVAL_INTERVAL = 250
EVAL_ITERS = 40

# data

text = Path("input.txt").read_text(encoding = "utf-8")

chars = sorted(set(text))
vocab_size = len(chars)

char_to_id = {
    char: index
    for index, char in enumerate(chars)
}

id_to_char = {
    index: char
    for char, index in char_to_id.items()
}

def encode(s):
    return [char_to_id[char] for char in s]

def decode(ids):
    return "".join(id_to_char[index] for index in ids)

data = torch.tensor(
    encode(text),
    dtype = torch.long,
)

#连续切分，避免验证集内容混入训练集
split_index = int(0.9 * len(data))

train_data = data[:split_index]
val_data = data[split_index:]

if len(val_data) <= BLOCK_SIZE:
    raise ValueError(
        "语料太短：验证集长度必须大于 BLOCK_SIZE"
    )

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def get_batch(split):
    source = train_data if split == "train" else val_data

    starts = torch.randint(
        low = 0,
        high = len(source) -  BLOCK_SIZE,
        size = (BATCH_SIZE,),
    )
    
    inputs = torch.stack(
        [
            source[start: start + BLOCK_SIZE]
            for start in starts
        ]
    )
    
    targets = torch.stack(
        [
            source[start + 1: start + BLOCK_SIZE + 1]
            for start in starts
        ]
    )
    
    return inputs.to(device), targets.to(device)

#estimate loss

@torch.no_grad()
def estimate_loss(model):
    previous_mode = model.training
    model.eval()
    
    results = {}

    # 每次调用都重新创建同种子的生成器。
    eval_generator = torch.Generator()
    eval_generator.manual_seed(2026)
    
    for split in ("train", "val"):
        losses = torch.empty(EVAL_ITERS)
        
        source = train_data if split == "train" else val_data
        
        for iteration in range(EVAL_ITERS):
            starts = torch.randint(
                low = 0,
                high = len(source) - BLOCK_SIZE,
                size = (BATCH_SIZE,),
                generator = eval_generator,
            )

            inputs = torch.stack([
                source[i:i + BLOCK_SIZE]
                for i in starts
            ]).to(device)

            targets = torch.stack([
                source[i + 1:i + BLOCK_SIZE + 1]
                for i in starts
            ]).to(device)

            _, loss = model(inputs, targets)
            losses[iteration] = loss.item()

        results[split] = losses.mean().item()
        
    model.train(previous_mode)
    return results

#generate

@torch.no_grad()
def generate(
    model,
    token_ids,
    max_new_tokens,
    *,
    greedy = False,
    temperature = 1.0,
    top_k = None,
):
    if not greedy and temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    
    previous_mode = model.training
    model.eval()
    
    for _ in range(max_new_tokens):
        #没有KV Cache，每生成一个字符都完全重算窗口
        context = token_ids[:, -model.max_seq_len:]
        
        logits, _ = model(context)
        logits = logits[:, -1, :]
        
        if greedy:
            next_token = torch.argmax(
                logits,
                dim = -1,
                keepdim = True
            )
        else:
            logits = logits / temperature
            
            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                
                top_values, _ = torch.topk(
                    logits,
                    k = k,
                    dim = -1,
                )
                
                threshold = top_values[:, -1:] #取每行的最后一列，同时保留维度（如果是[:, -1]就不保留）
                logits = logits.masked_fill(
                    logits < threshold,
                    float("-inf")
                )
            
            probabilities = torch.softmax(logits, dim = -1)
            
            next_token = torch.multinomial(
                probabilities,
                num_samples = 1
            )
        
        token_ids = torch.cat(
            [token_ids, next_token],
            dim = -1,
        )
    
    model.train(previous_mode)
    return token_ids

# generate cached

@torch.no_grad()
def generate_cached(
    model,
    token_ids,
    max_new_tokens,
    *,
    greedy = False,
    temperature = 1.0,
    top_k = None,
):
    if not greddy and temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    if token_ids.shape[i] + max_new_tokens > model.max_seq_len:
        raise ValueError(
            "简化版 KV Cache 暂时要求："
            "prompt长度 + 生成长度 <= max_seq_len"
        )
    
    previous_mode = model.training
    model.eval()
    
    #prefill: prompt一次并行处理，并建立每层缓存
    logits, _, cache = model(
        token_ids,
        use_cache = True,
    )
    
    for step in range(max_new_tokens):
        next_token_logits = logits[:, -1, :]
        
        if greedy:
            next_token = torch.argmax(
                next_token_logits,
                dim = -1,
                keepdim = True
            )
        else:
            next_token_logits = (
                next_token_logits / temperature
            )
            
            if top_k is not None:
                k = min(
                    top_k,
                    next_token_logits.shape[-1]
                )
                
                top_values, _ = torch.topk(
                    next_token_logits,
                    k = k,
                    dim = -1,
                )
                
                threshold = top_values[:, -1, :]
                
                next_token_logits = (
                    next_token_logits.masked_fill(
                        next_token_logits < threshold,
                        float("-inf"),
                    )
                )
                
            probabilities = torch.softmax(
                next_token_logits,
                dim = -1
            )
            
            next_token = torch.multinomial(
                probabilites,
                num_samples = 1
            )
            
        token_ids = torch.cat(
            [token_ids, next_token],
            dim = 1,
        )
        
        #最后一次采样之后，不必再计算logits
        if step + 1 < max_new_tokens:
            #这里只输入刚生成的一个token
            logits, _, cache = model(
                next_token,
                past_key_values = cache,
                use_cache = True
            )
        
        model.train(previous_mode)
        return token_ids

#train!

def main():
    torch.manual_seed(42)
    
    model = TinyGPT(
        vocab_size = vocab_size,
        d_model = D_MODEL,
        num_heads = NUM_HEADS,
        num_layers = NUM_LAYERS,
        max_seq_len = BLOCK_SIZE,
        dropout = DROPOUT,
    ).to(device)
    
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    
    print("device:", device)
    print("characters:", len(text))
    print("vocabulary size:", vocab_size)
    print("parameters:", parameter_count)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = MAX_LEARNING_RATE,
        weight_decay = 0.1,
    )
    
    best_val_loss = float("inf")
    
    for step in range(MAX_STEPS + 1):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)

            print(
                f"step {step:4d} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | "
                f"lr {get_learning_rate(step):.2e}"
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

                torch.save(
                    {
                        "step": step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                    },
                    "best_char_gpt.pt",
                )

                print("saved best checkpoint")

        # step=MAX_STEPS 时只评估，不再执行一次训练。
        if step == MAX_STEPS:
            break

        learning_rate = get_learning_rate(step)

        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        inputs, targets = get_batch("train")
        _, loss = model(inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        
    final_losses = estimate_loss(model)
    
    print(
        f"final     | "
        f"train {final_losses['train']:.4f} | "
        f"val {final_losses['val']:.4f}"
    )
    
    checkpoint = torch.load(
        "best_char_gpt.pt",
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state"])

    print(
        "loaded best checkpoint:",
        checkpoint["step"],
        checkpoint["val_loss"],
    )
    
    # 使用语料开头作为 prompt，确保字符都在词表中。
    prompt = text[:40]

    prompt_ids = torch.tensor(
        [encode(prompt)],
        dtype=torch.long,
        device=device,
    )

    experiments = [
        {
            "name": "greedy",
            "greedy": True,
        },
        {
            "name": "temperature=0.8, top_k=20",
            "temperature": 0.8,
            "top_k": 20,
        },
        {
            "name": "temperature=1.0, top_k=40",
            "temperature": 1.0,
            "top_k": 40,
        },
        {
            "name": "temperature=1.2, no top-k",
            "temperature": 1.2,
            "top_k": None,
        },
    ]

    for experiment in experiments:
        # 让采样实验可以复现。
        torch.manual_seed(1234)

        generated = generate(
            model,
            prompt_ids.clone(),
            max_new_tokens=400,
            greedy=experiment.get("greedy", False),
            temperature=experiment.get("temperature", 1.0),
            top_k=experiment.get("top_k"),
        )

        print(f"\n--- {experiment['name']} ---")
        print(decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
        