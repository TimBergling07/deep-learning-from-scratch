import torch
from torch import nn

from model import AlexNet
from dataset import train_loader, test_loader

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用设备:", device)

model = AlexNet(num_classes = 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.0001,
    weight_decay = 0.0005
)

num_epochs = 1

def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = logits.argmax(dim=1)
            total_correct += (
                predictions == labels
            ).sum().item()

    average_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return average_loss, accuracy

for epoch in range(num_epochs):
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_index, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)
        
        optimizer.zero_grad()
        
        logits = model(images)
        loss = criterion(logits, labels) #loss是平均损失(reduction = "mean")
        
        loss.backward()
        optimizer.step()
        
        batch_size = labels.size(0)
        
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        predictions = logits.argmax(dim = 1)
        total_correct += (
            predictions == labels
        ).sum().item()
        
        if(batch_index + 1) % 50 == 0:
            current_loss = total_loss / total_samples
            current_accuracy = (
                100.0 * total_correct / total_samples
            )
            
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Batch {batch_index + 1}/{len(train_loader)}, "
                f"Loss {current_loss:.4f}, "
                f"Accuracy {current_accuracy:.2f}%"
            )

    epoch_loss = total_loss / total_samples
    epoch_accuracy = 100.0 * total_correct / total_samples
    print(
        f"Epoch {epoch + 1} 完成："
        f"Loss={epoch_loss:.4f}, "
        f"Accuracy={epoch_accuracy:.2f}%"
    )
    
    test_loss, test_accuracy = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(
        f"测试集："
        f"Loss={test_loss:.4f}, "
        f"Accuracy={test_accuracy:.2f}%"
    )