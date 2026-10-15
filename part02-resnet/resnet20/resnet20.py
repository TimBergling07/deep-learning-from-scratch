import torch
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = stride,
            padding = 1,
            bias = False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias = False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace = False)

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity() #等价于一个什么也不做的模块
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = 1,
                    stride = stride,
                    bias = False,
                ),
                nn.BatchNorm2d(out_channels),
            )
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        assert out.shape == identity.shape, (
            f"main shape {out.shape} !="
            f"shortcut shape {identity.shape}"
        )
        
        out = out + identity
        out = self.relu(out)
                
        return out


def make_stage(
    in_channels,
    out_channels,
    num_blocks,
    first_stride,
):
    if num_blocks < 1:
        raise ValueError("num_blocks must be at least 1")

    blocks = []
        
    # 第一个block可能负责降采样和改变通道数
    blocks.append(
        BasicBlock(
            in_channels = in_channels,
            out_channels = out_channels,
            stride = first_stride,
        )
    )
        
    # 后续block保持通道数和空间尺寸不变
    for _ in range(1, num_blocks):
        blocks.append(
            BasicBlock(
                in_channels = out_channels,
                out_channels = out_channels,
                stride = 1,
            )
        )
        
    return nn.Sequential(*blocks) #*blocks是Python的序列解包

class ResNet20(nn.Module):
    def __init__(self, num_classes = 10, zero_init_residual = False, ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = False),
        )
        
        self.stage1 = make_stage(
            in_channels = 16,
            out_channels = 16,
            num_blocks = 3,
            first_stride = 1, 
        )
        
        self.stage2 = make_stage(
            in_channels = 16,
            out_channels = 32,
            num_blocks = 3,
            first_stride = 2,
        )
        
        self.stage3 = make_stage(
            in_channels = 32,
            out_channels = 64,
            num_blocks = 3,
            first_stride = 2,
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        self._initialize_weights(
            zero_init_residual = zero_init_residual
        )
    
    def _initialize_weights(self, zero_init_residual):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode = "fan_out",
                    nonlinearity = "relu",
                )
                
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
            elif isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean = 0.0,
                    std = 0.01,
                )
                nn.init.constant_(module.bias, 0)
        
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)
    
    def forward(self, x):
        out = self.stem(x)
        
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim = 1)
        out = self.fc(out)
        
        return out

def get_residual_output(block, x):
    out = block.conv1(x)
    out = block.bn1(out)
    out = block.relu(out)

    out = block.conv2(out)
    out = block.bn2(out)

    return out

if __name__ == "__main__":
    model = ResNet20(
        num_classes=10,
        zero_init_residual=False,
    )

    model.train()

    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 10, size=(8,))

    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, labels)

    loss.backward()

    missing_gradients = []
    invalid_gradients = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.grad is None:
            missing_gradients.append(name)
        elif not torch.isfinite(parameter.grad).all():
            invalid_gradients.append(name)

    print("loss:", loss.item())
    print("missing gradients:", missing_gradients)
    print("invalid gradients:", invalid_gradients)

    assert logits.shape == (8, 10)
    assert len(missing_gradients) == 0
    assert len(invalid_gradients) == 0