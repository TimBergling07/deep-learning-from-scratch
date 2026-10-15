import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from pathlib import Path


# 曲线必须读取 last_checkpoint，因为它包含完整 200 epochs history。
RESNET_LAST = Path(
    "./resnet20/checkpoints/last_checkpoint.pt"
)
RESNET_BEST = Path(
    "./resnet20/checkpoints/best_checkpoint.pt"
)

PLAINNET_LAST = Path(
    "./plainnet20/checkpoints/last_checkpoint.pt"
)
PLAINNET_BEST = Path(
    "./plainnet20/checkpoints/best_checkpoint.pt"
)


def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )


def percent(values):
    return [
        value * 100
        for value in values
    ]


def main():
    resnet_last = load_checkpoint(RESNET_LAST)
    resnet_best = load_checkpoint(RESNET_BEST)

    plainnet_last = load_checkpoint(PLAINNET_LAST)
    plainnet_best = load_checkpoint(PLAINNET_BEST)

    resnet_history = resnet_last["history"]
    plainnet_history = plainnet_last["history"]

    resnet_epochs = range(
        1,
        len(resnet_history["train_loss"]) + 1,
    )
    plainnet_epochs = range(
        1,
        len(plainnet_history["train_loss"]) + 1,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(17, 5),
    )

    axes[0].plot(
        resnet_epochs,
        resnet_history["train_loss"],
        label="ResNet Train",
        color="tab:blue",
    )
    axes[0].plot(
        resnet_epochs,
        resnet_history["validation_loss"],
        label="ResNet Validation",
        color="tab:blue",
        linestyle="--",
    )
    axes[0].plot(
        plainnet_epochs,
        plainnet_history["train_loss"],
        label="PlainNet Train",
        color="tab:orange",
    )
    axes[0].plot(
        plainnet_epochs,
        plainnet_history["validation_loss"],
        label="PlainNet Validation",
        color="tab:orange",
        linestyle="--",
    )

    axes[0].axvline(
        100,
        color="gray",
        linestyle=":",
        alpha=0.7,
    )
    axes[0].axvline(
        150,
        color="gray",
        linestyle=":",
        alpha=0.7,
    )

    axes[0].set_title("PlainNet-20 vs ResNet-20 Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        resnet_epochs,
        percent(
            resnet_history["train_accuracy"]
        ),
        label="ResNet Train",
        color="tab:blue",
    )
    axes[1].plot(
        resnet_epochs,
        percent(
            resnet_history[
                "validation_accuracy"
            ]
        ),
        label="ResNet Validation",
        color="tab:blue",
        linestyle="--",
    )
    axes[1].plot(
        plainnet_epochs,
        percent(
            plainnet_history["train_accuracy"]
        ),
        label="PlainNet Train",
        color="tab:orange",
    )
    axes[1].plot(
        plainnet_epochs,
        percent(
            plainnet_history[
                "validation_accuracy"
            ]
        ),
        label="PlainNet Validation",
        color="tab:orange",
        linestyle="--",
    )

    axes[1].axvline(
        100,
        color="gray",
        linestyle=":",
        alpha=0.7,
    )
    axes[1].axvline(
        150,
        color="gray",
        linestyle=":",
        alpha=0.7,
    )

    axes[1].set_title(
        "PlainNet-20 vs ResNet-20 Accuracy"
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()

    output_path = Path(
        "./plainnet_vs_resnet.png"
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(
        "ResNet best:",
        f"epoch {resnet_best['epoch']},",
        f"validation "
        f"{resnet_best['validation_accuracy'] * 100:.2f}%",
    )

    print(
        "PlainNet best:",
        f"epoch {plainnet_best['epoch']},",
        f"validation "
        f"{plainnet_best['validation_accuracy'] * 100:.2f}%",
    )

    print("saved:", output_path)


if __name__ == "__main__":
    main()