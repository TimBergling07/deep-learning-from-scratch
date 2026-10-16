import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from smallCNN import SmallCNN


CLASS_NAMES = (
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


def main():
    parser = argparse.ArgumentParser(
        description="Predict a single image with SmallCNN."
    )

    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to a PNG or JPG image.",
    )

    args = parser.parse_args()
    image_path = args.image_path.expanduser()

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("device:", device)
    print("image:", image_path.resolve())

    project_dir = Path(__file__).resolve().parent

    checkpoint_path = (
        project_dir
        / "checkpoints"
        / "best_checkpoint.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = SmallCNN().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    original_image = Image.open(image_path).convert("RGB")

    inference_transform = transforms.Compose([
        transforms.Resize(36),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    input_tensor = inference_transform(
        original_image
    ).unsqueeze(0).to(device)

    print("model input:", input_tensor.shape)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    top_probabilities, top_indices = (
        probabilities[0].topk(3)
    )

    predicted_index = top_indices[0].item()
    predicted_name = CLASS_NAMES[predicted_index]
    confidence = top_probabilities[0].item()

    print("\nTop-3 predictions:")

    for rank, (probability, class_index) in enumerate(
        zip(top_probabilities, top_indices),
        start=1,
    ):
        print(
            f"{rank}. "
            f"{CLASS_NAMES[class_index.item()]:10s} "
            f"{probability.item() * 100:6.2f}%"
        )

    figure, axis = plt.subplots(figsize=(5, 5))

    axis.imshow(original_image)
    axis.set_title(
        f"{predicted_name}: {confidence * 100:.2f}%"
    )
    axis.axis("off")

    figure.tight_layout()

    output_path = project_dir / "prediction_result.png"

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print("\nsaved result:", output_path)
    plt.show()


if __name__ == "__main__":
    main()