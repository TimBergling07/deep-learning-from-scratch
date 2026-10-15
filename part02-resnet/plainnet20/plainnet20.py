import torch
from torch import nn


class PlainBlock(nn.Module):
    """
    与 ResNet BasicBlock 的主分支保持一致，
    但完全移除 shortcut 和残差加法。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


def make_plain_stage(
    in_channels,
    out_channels,
    num_blocks,
    first_stride,
):
    if num_blocks < 1:
        raise ValueError(
            "num_blocks must be at least 1"
        )

    blocks = [
        PlainBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=first_stride,
        )
    ]

    for _ in range(1, num_blocks):
        blocks.append(
            PlainBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
            )
        )

    return nn.Sequential(*blocks)


class PlainNet20(nn.Module):
    """
    与当前 ResNet-20 保持相同的：
    - stem
    - 三个 stage
    - 每个 stage 三个双卷积 block
    - 通道数 16 -> 32 -> 64
    - 降采样位置
    - 池化与分类器

    唯一核心区别：没有 shortcut。
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
        )

        self.stage1 = make_plain_stage(
            in_channels=16,
            out_channels=16,
            num_blocks=3,
            first_stride=1,
        )

        self.stage2 = make_plain_stage(
            in_channels=16,
            out_channels=32,
            num_blocks=3,
            first_stride=2,
        )

        self.stage3 = make_plain_stage(
            in_channels=32,
            out_channels=64,
            num_blocks=3,
            first_stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(
                    module.weight,
                    1,
                )
                nn.init.constant_(
                    module.bias,
                    0,
                )

            elif isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.01,
                )
                nn.init.constant_(
                    module.bias,
                    0,
                )

    def forward(self, x):
        out = self.stem(x)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)

        return out


def test_model():
    model = PlainNet20(num_classes=10)
    model.eval()

    x = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        logits = model(x)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("input shape: ", x.shape)
    print("output shape:", logits.shape)
    print("parameters:  ", parameter_count)

    assert logits.shape == (4, 10)
    assert parameter_count == 269722

    print("PlainNet20 structure test passed.")


if __name__ == "__main__":
    test_model()