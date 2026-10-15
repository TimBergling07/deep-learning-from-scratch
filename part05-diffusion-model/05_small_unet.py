"""
Part 5 · Step 05：无 Attention 的小型 U-Net

输入：
    xt: [B, 1, 32, 32]
    t:  [B]，代码时间下标

输出：
    predicted_noise: [B, 1, 32, 32]

通道：
    32 -> 64 -> 128 -> 64 -> 32
"""

import importlib

import torch
from torch import nn
from torch.nn import functional as F


time_module = importlib.import_module("03_time_embedding")
block_module = importlib.import_module("04_time_resblock")

timestep_embedding = time_module.timestep_embedding
TimeMLP = block_module.TimeMLP
TimeResBlock = block_module.TimeResBlock


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_mlp = TimeMLP(
            embedding_dim=64,
            time_dim=128,
        )

        # 输入：[B, 1, 32, 32] -> [B, 32, 32, 32]
        self.input_conv = nn.Conv2d(
            1, 32, kernel_size=3, padding=1
        )
        self.enc1 = TimeResBlock(32, 32)

        # [B, 32, 32, 32] -> [B, 64, 16, 16]
        self.down1 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )
        self.enc2 = TimeResBlock(64, 64)

        # [B, 64, 16, 16] -> [B, 128, 8, 8]
        self.down2 = nn.Conv2d(
            64, 128, kernel_size=3, stride=2, padding=1
        )
        self.mid1 = TimeResBlock(128, 128)
        self.mid2 = TimeResBlock(128, 128)

        # [B, 128, 8, 8] -> [B, 64, 16, 16]
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
        )

        # 拼接 s2 后有 128 个通道。
        self.dec2 = TimeResBlock(128, 64)

        # [B, 64, 16, 16] -> [B, 32, 32, 32]
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
        )

        # 拼接 s1 后有 64 个通道。
        self.dec1 = TimeResBlock(64, 32)

        # 输出不限制数值范围。
        self.output_head = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, xt, t):
        # TODO 1：先计算 64 维时间编码，再经过 time_mlp。
        # time_emb 的尺寸应为 [B, 128]。
        time_emb = self.time_mlp(timestep_embedding(t, dim = 64))

        h = self.input_conv(xt)

        # 保存编码器特征，供解码器使用。
        s1 = self.enc1(h, time_emb)

        h = self.down1(s1)
        s2 = self.enc2(h, time_emb)

        h = self.down2(s2)
        h = self.mid1(h, time_emb)
        h = self.mid2(h, time_emb)

        h = self.up2(h)

        # TODO 2：沿通道维拼接 h 与 s2，h 放在前面。
        # 拼接后：[B, 128, 16, 16]
        h = torch.cat([h, s2], dim = 1)

        h = self.dec2(h, time_emb)
        h = self.up1(h)

        # TODO 3：沿通道维拼接 h 与 s1，h 放在前面。
        # 拼接后：[B, 64, 32, 32]
        h = torch.cat([h, s1], dim = 1)

        h = self.dec1(h, time_emb)

        return self.output_head(h)