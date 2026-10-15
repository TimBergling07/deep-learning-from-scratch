import torch
from torch import nn

def make_conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True),
        
        nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True),
        
        nn.MaxPool2d(kernel_size = 2, stride = 2),
    )

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.features = nn.Sequential(
            make_conv_block(3, 32),
            make_conv_block(32, 64),
            make_conv_block(64, 128),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) #从第一维开始展开 [N, 128, 1, 1] -> [N, 128]
        x = self.classifier(x)
        return x