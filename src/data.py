import csv
import random
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SIZE = (64, 192)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


def encode_label(label: str) -> list[int]:
    if len(label) != 6 or not label.isdigit():
        raise ValueError(f"label must be exactly 6 digits, got {label!r}")
    return [int(char) for char in label]


def decode_digits(digits: list[int]) -> str:
    if len(digits) != 6:
        raise ValueError("digits must contain exactly 6 values")
    return "".join(str(int(digit)) for digit in digits)


class RandomLightBlur:
    def __init__(self, probability: float = 0.18, max_radius: float = 0.7) -> None:
        self.probability = probability
        self.max_radius = max_radius

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, self.max_radius)))


def build_transform(train: bool) -> Callable[[Image.Image], torch.Tensor]:
    if train:
        return transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.18, contrast=0.22, saturation=0.08)], p=0.7),
                transforms.RandomAffine(degrees=3, translate=(0.03, 0.08), scale=(0.96, 1.04), shear=2),
                RandomLightBlur(),
                transforms.ToTensor(),
                transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


class ContainerDigitsDataset(Dataset):
    def __init__(
        self,
        root: Union[Path, str],
        split: str,
        augment: bool = False,
        max_samples: Optional[int] = None,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform if transform is not None else build_transform(train=augment)
        labels_path = self.root / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.csv not found: {labels_path}")

        rows = []
        with labels_path.open(newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                if row["split"] == split:
                    encode_label(row["label"])
                    rows.append(row)
        if max_samples is not None:
            rows = rows[:max_samples]
        if not rows:
            raise ValueError(f"no rows found for split {split!r}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        label = row["label"]
        image = Image.open(self.root / row["image"]).convert("RGB")
        target = torch.tensor(encode_label(label), dtype=torch.long)
        return self.transform(image), target, label
