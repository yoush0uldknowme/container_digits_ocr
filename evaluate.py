import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.data import ContainerDigitsDataset
from src.model import Fixed6DigitCNN
from src.train_utils import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fixed 6-digit container OCR model.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset_container_digits"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/first_fixed6/best_model.pt"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main(
    data_root: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    split: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> None:
    args = parse_args()
    if data_root is not None:
        args.data_root = data_root
    if checkpoint is not None:
        args.checkpoint = checkpoint
    if split is not None:
        args.split = split
    if batch_size is not None:
        args.batch_size = batch_size
    if num_workers is not None:
        args.num_workers = num_workers
    if max_samples is not None:
        args.max_samples = max_samples
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ContainerDigitsDataset(args.data_root, args.split, augment=False, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = Fixed6DigitCNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate_model(model, loader, device)
    print(f"split: {args.split}")
    print(f"loss: {metrics['loss']:.6f}")
    print(f"digit_acc: {metrics['digit_acc']:.6f}")
    print(f"sequence_acc: {metrics['sequence_acc']:.6f}")


if __name__ == "__main__":
    main()
