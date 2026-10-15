"""
Part 5 · Step 04：时间条件残差块

TimeMLP：
    [B, 64] -> [B, 128]

TimeResBlock：
    x:        [B, in_channels, H, W]
    time_emb: [B, time_dim]
    输出：    [B, out_channels, H, W]
"""

import importlib

import torch
from torch import nn
from torch.nn import functional as F


time_module = importlib.import_module("03_time_embedding")
timestep_embedding = time_module.timestep_embedding


class TimeMLP(nn.Module):
    def __init__(self, embedding_dim=64, time_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, embedding):
        return self.net(embedding)


class TimeResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_dim=128,
        groups=8,
    ):
        super().__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels 必须能被 groups 整除")
        if out_channels % groups != 0:
            raise ValueError("out_channels 必须能被 groups 整除")

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )

    def forward(self, x, time_emb):
        h = self.conv1(F.silu(self.norm1(x)))

        # 已得到每张图像、每个输出通道的时间偏置。
        time_bias = self.time_proj(F.silu(time_emb))

        # TODO 1：将 [B, out_channels] 变成
        # [B, out_channels, 1, 1]，便于空间广播。
        time_bias = time_bias[:, :, None, None]

        # TODO 2：先对 h 做 norm2，再加上 time_bias。
        h = self.norm2(h) + time_bias

        h = self.conv2(F.silu(h))

        # TODO 3：将 shortcut 分支与 h 相加。
        return self.skip(x) + h