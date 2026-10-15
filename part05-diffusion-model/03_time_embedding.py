"""
Part 5 · Step 03：正弦余弦时间编码

输入：
    t: [B]，每张图像的代码时间下标

输出：
    embedding: [B, dim]

约定：
    - dim 为正偶数，默认 64。
    - 时间下标不除以 T。
    - 前半部分为 sin，后半部分为 cos。
    - 频率 omega_i = 10000 ** (-i / half_dim)。
"""

import math

import torch


def timestep_embedding(t, dim=64):
    """将一批时间下标映射成正弦余弦向量。"""
    if dim < 2 or dim % 2 != 0:
        raise ValueError("dim 必须是大于等于 2 的偶数")

    if t.ndim != 1:
        raise ValueError("t 的尺寸必须是 [B]")

    half_dim = dim // 2

    # i = 0, 1, ..., half_dim-1
    indices = torch.arange(
        half_dim,
        dtype=torch.float32,
        device=t.device,
    )

    # TODO 1：按照公式构造频率，结果尺寸为 [half_dim]。
    # 可以使用：a ** x = exp(x * log(a))
    freqs = torch.exp(-indices / half_dim * math.log(10000.0))

    # 将整数时间下标转换为浮点数，参与三角函数计算。
    time = t.to(dtype=torch.float32)

    # TODO 2：计算所有时间与所有频率的相位。
    # 结果尺寸为 [B, half_dim]。
    angles = time[:, None] * freqs[None, :]

    # TODO 3：沿最后一维拼接 sin 和 cos。
    # 使用 torch.cat，结果尺寸为 [B, dim]。
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim = -1)

    return embedding