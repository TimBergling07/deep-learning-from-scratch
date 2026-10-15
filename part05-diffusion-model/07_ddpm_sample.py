"""
Part 5 · Step 07：DDPM 反向采样

读取已训练的 SmallUNet checkpoint。
从标准高斯噪声开始，逐步生成图像。

核心 TODO：
1. 后验方差。
2. 噪声预测参数化的反向均值。
3. 使用标准差加入随机噪声。
"""

import argparse
import importlib
import math
import time
from pathlib import Path

import torch
from torchvision.utils import save_image


model_module = importlib.import_module("05_small_unet")
SmallUNet = model_module.SmallUNet


def make_sampling_coefficients(betas):
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    # 数学中的：
    # [alpha_bar_0, alpha_bar_1, ..., alpha_bar_(T-1)]
    # 其中干净状态 alpha_bar_0 = 1。
    alpha_bar_prev = torch.cat([
        torch.ones_like(alpha_bar[:1]),
        alpha_bar[:-1],
    ])

    # TODO 1：计算后验方差，尺寸为 [T]。
    posterior_variance = betas *(1 - alpha_bar_prev) / (1 - alpha_bar)

    return alphas, alpha_bar, posterior_variance


def ddpm_step(
    xt,
    predicted_noise,
    i,
    betas,
    alphas,
    alpha_bar,
    posterior_variance,
    generator=None,
):
    # TODO 2：根据噪声预测计算反向均值。
    # 当前时间的系数分别是 betas[i]、alphas[i]、alpha_bar[i]。
    mean = (
        1.0 / torch.sqrt(alphas[i]) 
        * (
            xt - betas[i] / torch.sqrt(1 - alpha_bar[i])
            * predicted_noise
        )
    )

    # 最后一轮更新：不再抽取或加入随机噪声。
    if i == 0:
        return mean

    z = torch.randn(
        xt.shape,
        dtype=xt.dtype,
        device=xt.device,
        generator=generator,
    )

    # TODO 3：返回 mean 加上“标准差 × z”。
    # posterior_variance[i] 是方差。
    return mean + torch.sqrt(posterior_variance[i]) * z


def check_sampling_boundaries():
    """验证方差、时间边界，以及末步不再加入随机性。"""
    betas = torch.tensor([0.1, 0.2], dtype=torch.float32)
    alphas, alpha_bar, variance = make_sampling_coefficients(betas)

    # 两步调度的后验方差可以手算：
    # 第一步：0
    # 第二步：0.2 * (1 - 0.9) / (1 - 0.72) = 1/14
    expected_variance = torch.tensor([0.0, 1.0 / 14.0])
    torch.testing.assert_close(variance, expected_variance)

    assert torch.all(variance >= 0)
    assert torch.all(variance <= betas)

    # 使用已知的真实噪声，检查最后一步能否还原 x0。
    x0 = torch.full((2, 1, 4, 4), 0.8)
    generator = torch.Generator().manual_seed(42)

    noise = torch.randn(x0.shape, generator=generator)
    x1 = (
        alphas[0].sqrt() * x0
        + betas[0].sqrt() * noise
    )

    rng_before = generator.get_state().clone()

    recovered = ddpm_step(
        x1,
        noise,
        0,
        betas,
        alphas,
        alpha_bar,
        variance,
        generator=generator,
    )

    torch.testing.assert_close(
        recovered, x0, rtol=1e-5, atol=1e-5
    )
    assert torch.equal(rng_before, generator.get_state())

    print("后验方差与末步边界检查通过。")


@torch.no_grad()
def sample_ddpm(model, betas, num_samples, seed, device):
    model.eval()

    alphas, alpha_bar, variance = make_sampling_coefficients(betas)
    T = betas.shape[0]

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # 初始状态 x_T。
    x = torch.randn(
        (num_samples, 1, 32, 32),
        device=device,
        generator=generator,
    )

    for i in range(T - 1, -1, -1):
        # 批次内每张图像处于同一个时间步。
        t = torch.full(
            (num_samples,),
            i,
            dtype=torch.long,
            device=device,
        )

        predicted_noise = model(x, t)

        x = ddpm_step(
            x,
            predicted_noise,
            i,
            betas,
            alphas,
            alpha_bar,
            variance,
            generator=generator,
        )

        completed = T - i
        if completed % 100 == 0 or i == 0:
            if not torch.isfinite(x).all().item():
                raise RuntimeError(f"时间下标 {i} 出现 NaN 或 Inf")

            print(f"采样进度：{completed}/{T}", flush=True)

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples 必须大于等于 1")

    check_sampling_boundaries()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    base_dir = Path(__file__).resolve().parent

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = (
            base_dir / "outputs" / "ddpm_unconditional" / "best.pt"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint["model"])

    # 使用 checkpoint 保存的调度，不另建一套。
    betas = checkpoint["betas"].to(device)

    print(f"设备：{device}")
    print(f"checkpoint epoch：{checkpoint['epoch']}")
    print(f"验证损失：{checkpoint['history'][-1]['val_loss']:.6f}")
    print(f"样本数：{args.num_samples}，种子：{args.seed}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()

    samples = sample_ddpm(
        model,
        betas,
        args.num_samples,
        args.seed,
        device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    output_dir = base_dir / "outputs" / "ddpm_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    name = f"epoch{checkpoint['epoch']}_seed{args.seed}"

    # 保存未截断的原始结果，供之后分析。
    torch.save(
        {
            "samples": samples.cpu(),
            "seed": args.seed,
            "epoch": checkpoint["epoch"],
            "sampler": "ddpm",
        },
        output_dir / f"{name}.pt",
    )

    # 只在最终显示时映射与截断。
    display_samples = ((samples + 1.0) / 2.0).clamp(0.0, 1.0)

    image_path = output_dir / f"{name}.png"
    save_image(
        display_samples.cpu(),
        image_path,
        nrow=math.ceil(math.sqrt(args.num_samples)),
    )

    print(f"采样耗时：{elapsed:.2f}s")
    print(
        f"原始结果范围："
        f"[{samples.min().item():.4f}, {samples.max().item():.4f}]"
    )
    print(f"图像已保存到：{image_path.resolve()}")


if __name__ == "__main__":
    main()