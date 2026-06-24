import csv
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from reliable_inference import DEFAULT_WEIGHTS, load_models
from src.data import ContainerDigitsDataset


@torch.no_grad()
def collect(data_root: Path, split: str, checkpoints: Sequence[Path], weights: Sequence[float],
            batch_size: int, num_workers: int) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ContainerDigitsDataset(data_root, split, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    models = load_models(checkpoints, device)
    confidences, correct_sequences = [], []
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = sum(model(images) * weight for model, weight in zip(models, weights))
        probabilities = logits.softmax(dim=-1)
        predictions = probabilities.argmax(dim=-1)
        confidences.append(probabilities.max(dim=-1).values.min(dim=-1).values.cpu())
        correct_sequences.append(predictions.eq(targets).all(dim=1).cpu())
    return torch.cat(confidences), torch.cat(correct_sequences)


def evaluate_threshold(confidences: torch.Tensor, correct: torch.Tensor, threshold: float) -> Dict[str, float]:
    accepted = confidences >= threshold
    accepted_count = int(accepted.sum().item())
    accepted_correct = int((correct & accepted).sum().item())
    total = len(correct)
    return {
        "threshold": threshold,
        "coverage": accepted_count / total,
        "accepted_accuracy": accepted_correct / accepted_count if accepted_count else 1.0,
        "accepted_count": accepted_count,
        "review_count": total - accepted_count,
        "accepted_errors": accepted_count - accepted_correct,
    }


def choose_threshold(confidences: torch.Tensor, correct: torch.Tensor, target_accuracy: float):
    candidates = sorted(set([0.0, 1.0] + [round(float(value), 6) for value in confidences.tolist()]))
    rows = [evaluate_threshold(confidences, correct, threshold) for threshold in candidates]
    valid = [row for row in rows if row["accepted_count"] > 0 and row["accepted_accuracy"] >= target_accuracy]
    if not valid:
        return max(rows, key=lambda row: (row["accepted_accuracy"], row["coverage"])), rows
    return max(valid, key=lambda row: (row["coverage"], row["accepted_accuracy"])), rows


def main(data_root: Path, checkpoints: Sequence[Path], output_dir: Path,
         target_accuracy: float = 0.9999, batch_size: int = 192, num_workers: int = 0) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Calibrating confidence threshold on validation set...")
    val_confidence, val_correct = collect(data_root, "val", checkpoints, DEFAULT_WEIGHTS, batch_size, num_workers)
    selected, rows = choose_threshold(val_confidence, val_correct, target_accuracy)
    print(f"Selected threshold: {selected['threshold']:.6f}")
    print(f"Validation accepted accuracy: {selected['accepted_accuracy']:.6f}")
    print(f"Validation coverage: {selected['coverage']:.6f}")

    print("Evaluating fixed threshold on test set...")
    test_confidence, test_correct = collect(data_root, "test", checkpoints, DEFAULT_WEIGHTS, batch_size, num_workers)
    test_result = evaluate_threshold(test_confidence, test_correct, selected["threshold"])
    print(f"Test accepted accuracy: {test_result['accepted_accuracy']:.6f}")
    print(f"Test coverage: {test_result['coverage']:.6f}")
    print(f"Accepted: {test_result['accepted_count']}/10000")
    print(f"Manual review: {test_result['review_count']}/10000")
    print(f"Errors among accepted: {test_result['accepted_errors']}")

    config = {
        "threshold": selected["threshold"],
        "target_validation_accuracy": target_accuracy,
        "weights": list(DEFAULT_WEIGHTS),
        "validation": selected,
        "test": test_result,
    }
    (output_dir / "threshold.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "threshold_search.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"Threshold config saved to: {output_dir / 'threshold.json'}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    main(root / "dataset_container_digits",
         [root / "runs" / "v2_seed2" / "best_model.pt", root / "runs" / "v2_seed3" / "best_model.pt"],
         root / "analysis" / "reliable_inference")
