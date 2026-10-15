import torch

from model import AlexNet

model = AlexNet(num_classes = 1000)
model.eval()

actual_shapes = {}

def save_shape(name):
    def hook(module, inputs, output):
        actual_shapes[name] = tuple(output.shape)
        
    return hook #闭包，带捕获状态的lambda

layer_names = [
    "conv1",
    "pool1",
    "conv2",
    "pool2",
    "conv3",
    "conv4",
    "conv5",
    "pool5",
    "fc1",
    "fc2",
    "fc3",
]

for name in layer_names:
    layer = getattr(model, name)
    layer.register_forward_hook(save_shape(name))

x = torch.randn(2, 3, 227, 227)

with torch.inference_mode(): #这里只做推理，不记录反向传播所需的信息，可以减少内存占用。
    output = model(x)

expected_shapes = {
    "conv1": (2, 96, 55, 55),
    "pool1": (2, 96, 27, 27),
    "conv2": (2, 256, 27, 27),
    "pool2": (2, 256, 13, 13),
    "conv3": (2, 384, 13, 13),
    "conv4": (2, 384, 13, 13),
    "conv5": (2, 256, 13, 13),
    "pool5": (2, 256, 6, 6),
    "fc1": (2, 4096),
    "fc2": (2, 4096),
    "fc3": (2, 1000),
}

for name, expected_shape in expected_shapes.items():
    actual_shape = actual_shapes[name]
    print(f"{name:6s}: {actual_shape}")

    assert actual_shape == expected_shape, (
        f"{name} 尺寸错误："
        f"期望 {expected_shape}，实际 {actual_shape}"
    )

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
) #生成器表达式 实际上是 sum((p.numel() for p in model.parameters())) 然后两个()变成了一个←语法糖

assert tuple(output.shape) == (2, 1000)
assert parameter_count == 60_965_224

print(f"参数量: {parameter_count:,}")
print("全部结构测试通过！")