import csv
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import decode_digits
from src.metrics import compute_digit_and_sequence_accuracy


def sequence_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_digits = 0
    total_digit_correct = 0
    total_sequences = 0
    total_sequence_correct = 0

    for images, targets, _labels in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = sequence_loss(logits, targets)
        predictions = logits.argmax(dim=-1)
        correct_digits = predictions.eq(targets)
        total_loss += loss.item() * images.size(0)
        total_digits += targets.numel()
        total_digit_correct += correct_digits.sum().item()
        total_sequences += targets.size(0)
        total_sequence_correct += correct_digits.all(dim=1).sum().item()

    return {
        "loss": total_loss / max(total_sequences, 1),
        "digit_acc": total_digit_correct / max(total_digits, 1),
        "sequence_acc": total_sequence_correct / max(total_sequences, 1),
    }


def append_metrics(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_predictions(logits: torch.Tensor, labels: Iterable[str], limit: int = 5) -> str:
    predictions = logits.argmax(dim=-1).detach().cpu().tolist()
    pairs = []
    for prediction, label in list(zip(predictions, labels))[:limit]:
        pairs.append(f"{decode_digits(prediction)}/{label}")
    return " ".join(pairs)
