from pathlib import Path

import matplotlib.pyplot as plt
import torch

script_dir = Path(__file__).resolve().parent

checkpoint_path = (
    script_dir
    / "checkpoints"
    / "last_checkpoint.pt"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

history = checkpoint["history"]

epochs = range(
    1,
    len(history["train_loss"]) + 1,
)

train_accuracy = [
    accuracy * 100
    for accuracy in history["train_accuracy"]
]

validation_accuracy = [
    accuracy * 100
    for accuracy in history["validation_accuracy"]
]

figure, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 4),
)

# Loss 曲线
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

axes[0].set_title("Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].grid(alpha=0.3)
axes[0].legend()

# Accuracy 曲线
axes[1].plot(
    epochs,
    train_accuracy,
    label="Train Accuracy",
)

axes[1].plot(
    epochs,
    validation_accuracy,
    label="Validation Accuracy",
)

axes[1].set_title("Accuracy Curve")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].grid(alpha=0.3)
axes[1].legend()

figure.tight_layout()

output_path = (
    script_dir
    / "training_curves.png"
)

figure.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
)

print("saved figure:", output_path.resolve())

plt.show()