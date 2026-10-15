import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 96,
            kernel_size = 11,
            stride = 4,
            padding = 0,
        )
        self.relu1 = nn.ReLU(inplace = True) #inplace = True表示尽量直接复用原张量的内存，减少显存占用
        self.lrn1 = nn.LocalResponseNorm(
            size = 5,
            alpha = 5e-4,
            beta = 0.75,
            k = 2.0,
        ) #现在cnn很少用
        self.pool1 = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2
        )
        
        self.conv2 = nn.Conv2d(
            in_channels = 96,
            out_channels = 256,
            kernel_size = 5,
            stride = 1,
            padding = 2,
            groups = 2
        )
        self.relu2 = nn.ReLU(inplace = True)
        self.lrn2 = nn.LocalResponseNorm(
            size = 5,
            alpha = 5e-4,
            beta = 0.75,
            k = 2.0,
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
        )

        self.conv3 = nn.Conv2d(
            in_channels = 256,
            out_channels = 384,
            kernel_size = 3,
            stride = 1,
            padding = 1,
        )
        self.relu3 = nn.ReLU(inplace = True)
        
        self.conv4 = nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=2,
        )
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(
            in_channels=384,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=2,
        )
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
        )
        
        self.fc1 = nn.Linear(
            in_features = 256*6*6,
            out_features = 4096,
        )
        self.relu6 = nn.ReLU(inplace = True)
        self.dropout1 = nn.Dropout(p = 0.5)
           
        self.fc2 = nn.Linear(
            in_features = 4096,
            out_features = 4096,
        )
        self.relu7 = nn.ReLU(inplace = True)
        self.dropout2 = nn.Dropout(p = 0.5)
        
        self.fc3 = nn.Linear(
            in_features = 4096,
            out_features = num_classes,
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(
                    layer.weight,
                    mean = 0.0,
                    std = 0.01
                )
                nn.init.zeros_(layer.bias)
        
        layers_with_one_bias = (
            self.conv2,
            self.conv4,
            self.conv5,
            self.fc1,
            self.fc2,
        )
        
        for layer in layers_with_one_bias:
            nn.init.ones_(layer.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.lrn1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.lrn2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
    
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        
        x = torch.flatten(x, start_dim = 1)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x