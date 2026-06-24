import argparse
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.data import ContainerDigitsDataset
from src.model import Fixed6DigitCNN
from src.train_utils import append_metrics, evaluate_model, format_predictions, save_json, sequence_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fixed 6-digit container OCR model.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset_container_digits"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/first_fixed6"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--val-max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260527)
    return parser.parse_args()


def main(
    data_root: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    num_workers: Optional[int] = None,
    train_max_samples: Optional[int] = None,
    val_max_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    args = parse_args()
    if data_root is not None:
        args.data_root = data_root
    if run_dir is not None:
        args.run_dir = run_dir
    if epochs is not None:
        args.epochs = epochs
    if batch_size is not None:
        args.batch_size = batch_size
    if lr is not None:
        args.lr = lr
    if num_workers is not None:
        args.num_workers = num_workers
    if train_max_samples is not None:
        args.train_max_samples = train_max_samples
    if val_max_samples is not None:
        args.val_max_samples = val_max_samples
    if seed is not None:
        args.seed = seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ContainerDigitsDataset(args.data_root, "train", augment=True, max_samples=args.train_max_samples)
    val_dataset = ContainerDigitsDataset(args.data_root, "val", augment=False, max_samples=args.val_max_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = Fixed6DigitCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_sequence_acc = -1.0

    save_json(
        args.run_dir / "config.json",
        {
            "data_root": str(args.data_root),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": str(device),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "seed": args.seed,
        },
    )

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        last_logits = None
        last_labels = None

        for images, targets, labels in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = sequence_loss(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * images.size(0)
            train_count += images.size(0)
            last_logits = logits.detach()
            last_labels = labels

        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step(val_metrics["sequence_acc"])
        train_loss = train_loss_sum / max(train_count, 1)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_metrics['loss']:.6f}",
            "val_digit_acc": f"{val_metrics['digit_acc']:.6f}",
            "val_sequence_acc": f"{val_metrics['sequence_acc']:.6f}",
            "lr": f"{lr:.8f}",
            "seconds": f"{elapsed:.2f}",
        }
        append_metrics(args.run_dir / "metrics.csv", row)

        checkpoint_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        checkpoint = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_sequence_acc": val_metrics["sequence_acc"],
            "val_digit_acc": val_metrics["digit_acc"],
            "config": checkpoint_config,
        }
        torch.save(checkpoint, args.run_dir / "last_model.pt")
        if val_metrics["sequence_acc"] > best_sequence_acc:
            best_sequence_acc = val_metrics["sequence_acc"]
            torch.save(checkpoint, args.run_dir / "best_model.pt")

        sample_text = format_predictions(last_logits, last_labels) if last_logits is not None else ""
        print(
            f"epoch {epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_digit_acc={val_metrics['digit_acc']:.4f} "
            f"val_sequence_acc={val_metrics['sequence_acc']:.4f} "
            f"lr={lr:.6g} samples={sample_text}"
        )


if __name__ == "__main__":
    main()

