import torch
from torch import nn

from model import AlexNet
from dataset import train_loader


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = AlexNet(num_classes=10).to(device)
model.train()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005,
)

images, labels = next(iter(train_loader))

images = images.to(device, non_blocking=True)
labels = labels.to(device, non_blocking=True)

for step in range(100):
    optimizer.zero_grad()

    logits = model(images)
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()

    predictions = logits.argmax(dim=1)
    accuracy = (
        predictions == labels
    ).float().mean().item() * 100

    if step == 0 or (step + 1) % 10 == 0:
        print(
            f"Step {step + 1:3d}, "
            f"Loss={loss.item():.4f}, "
            f"Accuracy={accuracy:.2f}%"
        )
