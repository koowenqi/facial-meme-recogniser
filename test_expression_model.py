from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from expression_model import ExpressionCNN


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ExpressionTestDataset(Dataset):
    def __init__(self, root: Path, class_names: list[str], image_size: int) -> None:
        self.root = root
        self.class_names = class_names
        self.class_to_index = {name: index for index, name in enumerate(class_names)}
        self.samples: list[tuple[Path, int]] = []
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        if not root.exists():
            raise RuntimeError(f"Test dataset folder does not exist: {root}")

        for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            label = label_dir.name.lower()
            if label not in self.class_to_index:
                print(f"Skipping test label not used by model: {label_dir.name}")
                continue

            for image_path in label_dir.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, self.class_to_index[label]))

        if not self.samples:
            raise RuntimeError(f"No test images found in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label_index = self.samples[index]
        image = Image.open(image_path).convert("L")
        return self.transform(image), label_index


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def test_model(model_path: Path, test_dir: Path, batch_size: int) -> None:
    checkpoint = torch.load(model_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 48)

    dataset = ExpressionTestDataset(test_dir, class_names, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExpressionCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    confusion = torch.zeros((len(class_names), len(class_names)), dtype=torch.int64)
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)

            total_correct += (predictions == labels).sum().item()
            total_seen += labels.size(0)
            for actual, predicted in zip(labels.cpu(), predictions.cpu()):
                confusion[actual, predicted] += 1

    print(f"Test images: {total_seen}")
    print(f"Accuracy: {total_correct / total_seen:.3f}")
    print("\nPer-class accuracy:")
    for index, class_name in enumerate(class_names):
        class_total = confusion[index].sum().item()
        class_correct = confusion[index, index].item()
        accuracy = class_correct / class_total if class_total else 0.0
        print(f"  {class_name}: {accuracy:.3f} ({class_correct}/{class_total})")

    print("\nConfusion matrix rows=actual columns=predicted")
    print("labels:", ", ".join(class_names))
    for index, class_name in enumerate(class_names):
        values = " ".join(str(value.item()).rjust(5) for value in confusion[index])
        print(f"{class_name.rjust(8)} {values}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a trained facial-expression model.")
    parser.add_argument("--model", default="models/expression_model.pt", help="Path to trained .pt model.")
    parser.add_argument("--test-dir", default="datasets/FER2013/test", help="Folder with test expression subfolders.")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_model(resolve_path(args.model), resolve_path(args.test_dir), args.batch_size)


if __name__ == "__main__":
    Image.init()
    main()
