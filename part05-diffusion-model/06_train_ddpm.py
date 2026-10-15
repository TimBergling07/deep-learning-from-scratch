"""
Part 5 · Step 06：无条件 DDPM 正式训练

复用：
    01_forward_diffusion.py
    05_small_unet.py

训练：
    每次重新抽取时间下标和噪声。

验证：
    每次评估使用相同的图像、时间下标与噪声。
"""

import argparse
import importlib
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


forward_module = importlib.import_module("01_forward_diffusion")
model_module = importlib.import_module("05_small_unet")

make_schedule = forward_module.make_schedule
q_sample = forward_module.q_sample
SmallUNet = model_module.SmallUNet


def make_loaders(data_dir, batch_size, workers, device):
    transform = transforms.Compose([
        transforms.Pad(2, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    dataset = datasets.FashionMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )

    train_set, val_set = random_split(
        dataset,
        [54_000, 6_000],
        generator=torch.Generator().manual_seed(42),
    )

    common = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        **common,
    )

    # 固定验证 batch size，使改变训练 batch size 时，
    # 验证中的随机数分配方式仍保持一致。
    val_loader = DataLoader(
        val_set,
        batch_size=64,
        shuffle=False,
        **common,
    )

    return train_loader, val_loader


def diffusion_loss(model, x0, alpha_bar, generator=None):
    B = x0.shape[0]
    T = alpha_bar.shape[0]

    t = torch.randint(0, T, (B, ), device = x0.device, generator = generator)

    noise = torch.randn(
        x0.shape,
        dtype=x0.dtype,
        device=x0.device,
        generator=generator,
    )

    xt, _ = q_sample(x0, t, alpha_bar, noise)

    predicted_noise = model(xt, t)

    loss = F.mse_loss(predicted_noise, noise)

    return loss


def train_one_epoch(model, loader, optimizer, alpha_bar, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x0, _ in loader:
        x0 = x0.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = diffusion_loss(model, x0, alpha_bar)

        loss_value = loss.item()
        if not torch.isfinite(loss).item():
            raise RuntimeError("训练损失出现 NaN 或 Inf")

        loss.backward()
        optimizer.step()

        B = x0.shape[0]
        total_loss += loss_value * B
        total_samples += B

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, alpha_bar, device):
    model.eval()

    # 每次评估重新从相同种子开始。
    # 使用独立 generator，不重置训练使用的随机数状态。
    generator = torch.Generator(device=device)
    generator.manual_seed(2026)

    total_loss = 0.0
    total_samples = 0

    for x0, _ in loader:
        x0 = x0.to(device, non_blocking=True)

        loss = diffusion_loss(
            model,
            x0,
            alpha_bar,
            generator=generator,
        )

        loss_value = loss.item()
        if not torch.isfinite(loss).item():
            raise RuntimeError("验证损失出现 NaN 或 Inf")

        B = x0.shape[0]
        total_loss += loss_value * B
        total_samples += B

    return total_loss / total_samples


def save_curve(history, path):
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, label="Train")
    ax.plot(epochs, val_losses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Noise prediction MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs" / "ddpm_unconditional"
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"

    config = {
        "T": 1000,
        "learning_rate": 2e-4,
        "batch_size": args.batch_size,
        "seed": 42,
        "prediction_target": "epsilon",
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "image_size": 32,
    }

    train_loader, val_loader = make_loaders(
        base_dir / "data",
        args.batch_size,
        args.workers,
        device,
    )

    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    betas, _, alpha_bar = make_schedule(
        T=config["T"],
        device=device,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    history = []

    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint：{latest_path}")

        checkpoint = torch.load(
            latest_path,
            map_location="cpu",
            weights_only=True,
        )

        # 防止恢复时误用另一套扩散调度。
        torch.testing.assert_close(
            checkpoint["betas"],
            betas.cpu(),
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        history = checkpoint["history"]

        # 恢复主进程使用的随机数状态。
        torch.set_rng_state(checkpoint["cpu_rng_state"])
        if device.type == "cuda":
            cuda_states = checkpoint["cuda_rng_states"]
            if cuda_states:
                torch.cuda.set_rng_state_all(cuda_states)

        print(f"已恢复，下一轮为 epoch {start_epoch}")

    print(f"设备：{device}")
    print(f"训练样本：{len(train_loader.dataset)}")
    print(f"验证样本：{len(val_loader.dataset)}")
    print(f"训练 batch size：{args.batch_size}")
    print(f"保存目录：{output_dir.resolve()}")

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            alpha_bar,
            device,
        )

        val_loss = evaluate(
            model,
            val_loader,
            alpha_bar,
            device,
        )

        elapsed = time.perf_counter() - started
        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
            "config": config,
            "betas": betas.detach().cpu(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else []
            ),
        }

        torch.save(checkpoint, latest_path)
        if improved:
            torch.save(checkpoint, best_path)

        save_curve(history, output_dir / "loss_curve.png")

        marker = " | best" if improved else ""
        print(
            f"epoch {epoch:3d}/{args.epochs} | "
            f"train {train_loss:.6f} | "
            f"val {val_loss:.6f} | "
            f"{elapsed:.1f}s"
            f"{marker}",
            flush=True,
        )


if __name__ == "__main__":
    main()