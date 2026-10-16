from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from smallcnn import SmallCNN

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
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device:", device)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    checkpoint_path = Path(
        "./checkpoints/best_checkpoint.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    # 这是一个新创建、初始参数随机f的模型
    loaded_model = SmallCNN().to(device)

    load_result = loaded_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("load result:", load_result)
    print("saved epoch:", checkpoint["epoch"])
    print(
        "saved validation accuracy:",
        f'{checkpoint["validation_accuracy"] * 100:.2f}%',
    )

    criterion = nn.CrossEntropyLoss()

    test_loss, test_accuracy = evaluate(
        model=loaded_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(
        f"loaded model | "
        f"test loss {test_loss:.4f} | "
        f"test accuracy {test_accuracy * 100:.2f}%"
    )
    
    class_names = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    
    image_index = 0
    
    image, true_label = test_dataset[image_index]
    print("single image shape:", image.shape)
    print("true label:", class_names[true_label])
    
    input_tensor = image.unsqueeze(0).to(device)
    
    print("model input shape:", input_tensor.shape)
    
    loaded_model.eval()
    
    with torch.inference_mode():
        logits = loaded_model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
    
    predicted_label = probabilities.argmax(dim = 1).item()
    confidence = probabilities[0, predicted_label].item()
    
    print("predicted:", class_names[predicted_label])
    print(f"confidence: {confidence * 100:.2f}%")
    print("correct:", predicted_label == true_label)
    
    top_probabilities, top_indices = probabilities[0].topk(3)

    print("top-3 predictions:")

    for rank, (probability, class_index) in enumerate(
        zip(top_probabilities, top_indices),
        start=1,
    ):
        print(
            f"{rank}. "
            f"{class_names[class_index.item()]:10s} "
            f"{probability.item() * 100:6.2f}%"
        )

if __name__ == "__main__":
    main()