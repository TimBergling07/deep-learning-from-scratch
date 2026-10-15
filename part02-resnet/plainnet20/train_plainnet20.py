import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from plainnet20 import PlainNet20


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(
        data_loader,
        desc="Training",
    ):
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss: {loss.item()}"
            )

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    average_accuracy = total_correct / total_samples

    return average_loss, average_accuracy


def evaluate(
    model,
    data_loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    average_accuracy = total_correct / total_samples

    return average_loss, average_accuracy


def save_curves(history, output_path):
    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 5),
    )

    axes[0].plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
    )
    axes[0].plot(
        epochs,
        history["validation_loss"],
        label="Validation Loss",
    )
    axes[0].set_title("PlainNet-20 Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        [
            accuracy * 100
            for accuracy in history["train_accuracy"]
        ],
        label="Train Accuracy",
    )
    axes[1].plot(
        epochs,
        [
            accuracy * 100
            for accuracy
            in history["validation_accuracy"]
        ],
        label="Validation Accuracy",
    )
    axes[1].set_title("PlainNet-20 Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)

    checkpoint_dir = Path(
        "./checkpoints"
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(
            size=32,
            padding=4,
            padding_mode="reflect",
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_base_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )

    validation_base_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=eval_transform,
    )

    split_generator = (
        torch.Generator().manual_seed(42)
    )

    all_indices = torch.randperm(
        len(train_base_dataset),
        generator=split_generator,
    ).tolist()

    validation_size = 5000

    validation_indices = all_indices[
        :validation_size
    ]
    train_indices = all_indices[
        validation_size:
    ]

    train_dataset = Subset(
        train_base_dataset,
        train_indices,
    )

    validation_dataset = Subset(
        validation_base_dataset,
        validation_indices,
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=eval_transform,
    )

    assert len(train_dataset) == 45000
    assert len(validation_dataset) == 5000
    assert len(test_dataset) == 10000

    assert set(train_indices).isdisjoint(
        validation_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=2,
    )

    print("training samples:  ", len(train_dataset))
    print(
        "validation samples:",
        len(validation_dataset),
    )
    print("test samples:      ", len(test_dataset))

    model = PlainNet20(
        num_classes=10
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("parameters:", parameter_count)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
    )

    num_epochs = 200

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=0.1,
    )

    best_validation_accuracy = -1.0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "learning_rate": [],
    }

    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(
            train_accuracy
        )
        history["validation_loss"].append(
            validation_loss
        )
        history["validation_accuracy"].append(
            validation_accuracy
        )
        history["learning_rate"].append(current_lr)

        print(
            f"epoch {epoch:3d}/{num_epochs} | "
            f"lr {current_lr:.4f} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy * 100:6.2f}% | "
            f"val loss {validation_loss:.4f} | "
            f"val acc {validation_accuracy * 100:6.2f}%"
        )

        is_best = (
            validation_accuracy
            > best_validation_accuracy
        )

        if is_best:
            best_validation_accuracy = (
                validation_accuracy
            )

        scheduler.step()

        checkpoint = {
            "architecture": "PlainNet20",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict(),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy":
                validation_accuracy,
            "best_validation_accuracy":
                best_validation_accuracy,
            "history": history,
            "config": {
                "num_epochs": num_epochs,
                "batch_size": 128,
                "optimizer": "SGD",
                "initial_learning_rate": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "milestones": [100, 150],
                "gamma": 0.1,
                "split_seed": 42,
            },
        }

        torch.save(
            checkpoint,
            checkpoint_dir
            / "last_checkpoint.pt",
        )

        if is_best:
            torch.save(
                checkpoint,
                checkpoint_dir
                / "best_checkpoint.pt",
            )

            print(
                "  saved new best checkpoint: "
                f"{best_validation_accuracy * 100:.2f}%"
            )

    print("\nTraining finished.")
    print(
        "Loading the best validation checkpoint..."
    )

    best_checkpoint = torch.load(
        checkpoint_dir / "best_checkpoint.pt",
        map_location=device,
        weights_only=True,
    )

    best_model = PlainNet20(
        num_classes=10
    ).to(device)

    best_model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    final_test_loss, final_test_accuracy = evaluate(
        model=best_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(
        "best validation epoch:",
        best_checkpoint["epoch"],
    )
    print(
        "best validation accuracy:",
        f"{best_checkpoint['validation_accuracy'] * 100:.2f}%",
    )
    print(
        f"final test | "
        f"loss {final_test_loss:.4f} | "
        f"accuracy {final_test_accuracy * 100:.2f}%"
    )

    save_curves(
        history=history,
        output_path=checkpoint_dir
        / "plainnet20_curves.png",
    )

    print(
        "saved curves:",
        checkpoint_dir
        / "plainnet20_curves.png",
    )


if __name__ == "__main__":
    main()