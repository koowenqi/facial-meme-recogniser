from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from expression_model import ExpressionCNN


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABEL_ALIASES = {
    "anger": "angry",
    "sadness": "sad",
}


@dataclass(frozen=True)
class TrainConfig:
    train_dirs: list[str]
    output: str
    image_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    validation_split: float
    seed: int


class MultiFolderExpressionDataset(Dataset):
    def __init__(self, roots: list[Path], class_names: list[str], transform) -> None:
        self.class_names = class_names
        self.class_to_index = {name: index for index, name in enumerate(class_names)}
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for root in roots:
            if not root.exists():
                print(f"Skipping missing dataset folder: {root}")
                continue

            for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                label = normalize_label(label_dir.name)
                if label not in self.class_to_index:
                    print(f"Skipping unsupported label '{label_dir.name}' in {root}")
                    continue

                label_index = self.class_to_index[label]
                for image_path in label_dir.rglob("*"):
                    if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                        self.samples.append((image_path, label_index))

        if not self.samples:
            raise RuntimeError("No training images found. Check your --train-dirs paths.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label_index = self.samples[index]
        image = Image.open(image_path).convert("L")
        return self.transform(image), label_index


def normalize_label(label: str) -> str:
    return LABEL_ALIASES.get(label.lower(), label.lower())


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def discover_classes(roots: list[Path]) -> list[str]:
    labels: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for label_dir in root.iterdir():
            if label_dir.is_dir():
                labels.add(normalize_label(label_dir.name))

    if len(labels) < 2:
        raise RuntimeError("Need at least 2 expression labels to train.")
    return sorted(labels)


def build_loaders(config: TrainConfig) -> tuple[DataLoader, DataLoader, list[str]]:
    roots = [resolve_path(path) for path in config.train_dirs]
    class_names = discover_classes(roots)

    train_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    full_dataset = MultiFolderExpressionDataset(roots, class_names, train_transform)
    val_source = MultiFolderExpressionDataset(roots, class_names, eval_transform)

    indices = list(range(len(full_dataset)))
    random.Random(config.seed).shuffle(indices)
    val_size = max(1, int(len(indices) * config.validation_split))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(val_source, val_indices), batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, class_names


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size

    return total_loss / total_seen, total_correct / total_seen


def train(config: TrainConfig) -> None:
    train_loader, val_loader, class_names = build_loaders(config)
    output = resolve_path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExpressionCNN(num_classes=len(class_names)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    best_val_accuracy = 0.0

    print(f"Classes: {', '.join(class_names)}")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Device: {device}")

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, loss_fn, device, optimizer)
        with torch.no_grad():
            val_loss, val_accuracy = run_epoch(model, val_loader, loss_fn, device)

        print(
            f"epoch {epoch:02d}/{config.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}"
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "image_size": config.image_size,
                    "config": asdict(config),
                    "best_val_accuracy": best_val_accuracy,
                    "model_type": "ExpressionCNN",
                },
                output,
            )

    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "class_names": class_names,
                "best_val_accuracy": best_val_accuracy,
                "output": str(output),
                "config": asdict(config),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved best model to: {output}")
    print(f"Best validation accuracy: {best_val_accuracy:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch facial-expression model.")
    parser.add_argument(
        "--train-dirs",
        nargs="+",
        default=["datasets/FER2013/train", "datasets/ck+"],
        help="Training folders with expression subfolders.",
    )
    parser.add_argument("--output", default="models/expression_model.pt", help="Model output path.")
    parser.add_argument("--image-size", type=int, default=48, help="Input image size. FER2013 images are 48x48.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        TrainConfig(
            train_dirs=args.train_dirs,
            output=args.output,
            image_size=args.image_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_split=args.validation_split,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    Image.init()
    main()
