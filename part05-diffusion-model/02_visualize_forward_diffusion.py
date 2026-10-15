"""
Part 5 · Step 02：观察 Fashion-MNIST 的前向加噪

处理顺序：
    28×28 原图 -> 黑色补边到 32×32 -> [0,1] -> [-1,1]

每类选择一张图像，使用同一份噪声比较不同时间步。
"""

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


# 原文件名以数字开头，使用 importlib 加载。
# 加载模块不会执行它的 if __name__ == "__main__" 部分。
forward = importlib.import_module("01_forward_diffusion")
make_schedule = forward.make_schedule
q_sample = forward.q_sample


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_examples(data_dir):
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

    # 字典：类别编号 -> 该类别第一张图像。
    # 类比 C++ 的 map<int, Tensor>。
    examples = {}

    for image, label in dataset:
        if label not in examples:
            examples[label] = image

        if len(examples) == 10:
            break

    # 按类别 0～9 排列，新增 batch 维度。
    x0 = torch.stack([examples[label] for label in range(10)])

    # 这些检查针对本轮的数据处理约定。
    assert x0.shape == (10, 1, 32, 32)
    assert x0.min().item() >= -1.0
    assert x0.max().item() <= 1.0

    # 黑色补边归一化后应为 -1。
    assert torch.all(x0[:, :, :2, :] == -1)
    assert torch.all(x0[:, :, -2:, :] == -1)
    assert torch.all(x0[:, :, :, :2] == -1)
    assert torch.all(x0[:, :, :, -2:] == -1)

    return x0


def plot_forward_diffusion(x0, output_path):
    torch.manual_seed(42)

    T = 1000
    _, _, alpha_bar = make_schedule(T=T)

    # 每张图像各有一份噪声；不同时间步复用它。
    noise = torch.randn_like(x0)

    # 使用数学时间步编号：0 是干净图，1000 是终点。
    steps = [0, 50, 100, 250, 500, 750, 1000]
    batches = []

    print("数学时间步 | 信号系数 | 噪声标准差")

    for step in steps:
        if step == 0:
            xt = x0
            signal_scale = 1.0
            noise_std = 0.0
        else:
            # 数学第 step 步 -> Python 数组下标 step-1。
            t = torch.full(
                (x0.shape[0],),
                step - 1,
                dtype=torch.long,
            )

            xt, _ = q_sample(
                x0, t, alpha_bar, noise=noise
            )

            signal_scale = alpha_bar[step - 1].sqrt().item()
            noise_std = (1.0 - alpha_bar[step - 1]).sqrt().item()

        batches.append(xt)

        print(
            f"{step:>10} | "
            f"{signal_scale:.6f} | "
            f"{noise_std:.6f}"
        )

    fig, axes = plt.subplots(
        nrows=10,
        ncols=len(steps),
        figsize=(12, 15),
    )

    for row in range(10):
        for col, step in enumerate(steps):
            xt = batches[col][row, 0]

            # 仅为显示映射回 [0,1] 并截断。
            # 不修改原始带噪张量。
            display_image = ((xt + 1.0) / 2.0).clamp(0.0, 1.0)

            ax = axes[row, col]
            ax.imshow(
                display_image.numpy(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(f"t = {step}")

            if col == 0:
                ax.set_ylabel(CLASS_NAMES[row], fontsize=9)

    fig.suptitle(
        "Fashion-MNIST: forward diffusion with fixed noise",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"\n图像已保存到：{output_path.resolve()}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    data_dir = base_dir / "data"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    x0 = load_examples(data_dir)

    plot_forward_diffusion(
        x0,
        output_dir / "02_forward_diffusion.png",
    )