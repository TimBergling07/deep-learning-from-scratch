"""
Part 5 · Step 01：扩散模型的前向加噪

学习目标：
1. 实现线性 beta 调度与累计系数 alpha_bar。
2. 实现批量前向加噪。
3. 验证累计系数，以及加噪后的均值、方差和标准差。

时间索引约定：
- x0 表示干净图像。
- 代码 t=0 对应数学中的第一次加噪。
- alpha_bar[0] 对应数学中的 alpha_bar_1，而不是干净状态。
"""

import torch


def make_schedule(T=1000, device="cpu"):
    """生成调度；三个返回张量的尺寸均为 [T]。"""
    betas = torch.linspace(
        1e-4,
        0.02,
        steps=T,
        dtype=torch.float32,
        device=device,
    )
    alphas = 1.0 - betas

    alpha_bar = torch.cumprod(alphas, dim=0, dtype=torch.float32)

    return betas, alphas, alpha_bar


def q_sample(x0, t, alpha_bar, noise=None):
    """
    从干净图像直接构造指定时间步的带噪图像。

    参数：
        x0:        [B, C, H, W]，浮点张量
        t:         [B]，torch.long，取值范围 [0, T-1]
        alpha_bar: [T]，累计系数表
        noise:     与 x0 同尺寸的标准高斯噪声；省略时自动抽取

    所有输入张量应位于同一设备。

    返回：
        xt:    与 x0 同尺寸的带噪图像
        noise: 本次使用的标准高斯噪声
    """
    if noise is None:
        noise = torch.randn_like(x0)

    # 每张图像取出一个系数，并广播到其全部通道与像素。
    a_bar = alpha_bar[t].reshape(-1, 1, 1, 1)

    xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

    return xt, noise