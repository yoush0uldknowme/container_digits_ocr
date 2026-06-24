import csv
import itertools
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from src.data import ContainerDigitsDataset, decode_digits
from src.model_v2 import Fixed6DigitCRNN


@torch.no_grad()
def collect_logits(data_root: Path, split: str, checkpoints: Sequence[Path], batch_size: int,
                   num_workers: int) -> Tuple[torch.Tensor, List[torch.Tensor], List[str], List[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ContainerDigitsDataset(data_root, split, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    models = []
    for checkpoint_path in checkpoints:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = Fixed6DigitCRNN().to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)

    targets_all, model_logits, labels_all = [], [[] for _ in models], []
    for images, targets, labels in loader:
        images = images.to(device, non_blocking=True)
        targets_all.append(targets)
        labels_all.extend(labels)
        for index, model in enumerate(models):
            model_logits[index].append(model(images).float().cpu())
    paths = [row["image"] for row in dataset.rows]
    return torch.cat(targets_all), [torch.cat(parts) for parts in model_logits], labels_all, paths


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(targets)
    sequence_correct = correct.all(dim=1)
    return {
        "digit_acc": correct.float().mean().item(),
        "sequence_acc": sequence_correct.float().mean().item(),
        "correct_sequences": int(sequence_correct.sum().item()),
    }


def weight_candidates() -> List[Tuple[float, float, float]]:
    candidates = []
    for a in range(11):
        for b in range(11 - a):
            c = 10 - a - b
            candidates.append((a / 10.0, b / 10.0, c / 10.0))
    return candidates


def weighted_logits(logits: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    return sum(logit * weight for logit, weight in zip(logits, weights))


def select_weights(logits: Sequence[torch.Tensor], targets: torch.Tensor):
    rows = []
    best_weights = (1 / 3, 1 / 3, 1 / 3)
    best_key = (-1.0, -1.0)
    for weights in weight_candidates():
        result = calculate_metrics(weighted_logits(logits, weights), targets)
        row = {"seed1_weight": weights[0], "seed2_weight": weights[1],
               "seed3_weight": weights[2], **result}
        rows.append(row)
        key = (result["sequence_acc"], result["digit_acc"])
        if key > best_key:
            best_key = key
            best_weights = weights
    return best_weights, rows


def export_errors(data_root: Path, output_dir: Path, targets: torch.Tensor,
                  logits: Sequence[torch.Tensor], paths: Sequence[str], weights: Sequence[float]) -> None:
    wrong_dir = output_dir / "wrong_images"
    wrong_dir.mkdir(parents=True, exist_ok=True)
    predictions = [item.argmax(dim=-1) for item in logits]
    ensemble = weighted_logits(logits, weights).argmax(dim=-1)
    rows = []
    for index, target in enumerate(targets):
        truth = decode_digits(target.tolist())
        individual = [decode_digits(pred[index].tolist()) for pred in predictions]
        combined = decode_digits(ensemble[index].tolist())
        if combined != truth:
            source = data_root / paths[index]
            filename = (f"{index:05d}_true-{truth}_s1-{individual[0]}_"
                        f"s2-{individual[1]}_s3-{individual[2]}_ens-{combined}{source.suffix}")
            shutil.copy2(source, wrong_dir / filename)
            rows.append({"index": index, "image": paths[index], "true": truth,
                         "seed1": individual[0], "seed2": individual[1],
                         "seed3": individual[2], "ensemble": combined})
    with (output_dir / "errors.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["index", "image", "true", "seed1", "seed2", "seed3", "ensemble"])
        writer.writeheader()
        writer.writerows(rows)


def main(data_root: Path, checkpoints: Sequence[Path], output_dir: Path,
         batch_size: int = 192, num_workers: int = 0) -> None:
    if len(checkpoints) != 3:
        raise ValueError("Exactly three V2 checkpoints are required")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1/3: validation prediction and weight selection")
    val_targets, val_logits, _, _ = collect_logits(data_root, "val", checkpoints, batch_size, num_workers)
    weights, search_rows = select_weights(val_logits, val_targets)
    with (output_dir / "weight_search.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(search_rows[0].keys()))
        writer.writeheader()
        writer.writerows(search_rows)
    print(f"Selected weights on validation: {weights}")

    print("Step 2/3: final test evaluation")
    targets, logits, _, paths = collect_logits(data_root, "test", checkpoints, batch_size, num_workers)
    individual_results = [calculate_metrics(item, targets) for item in logits]
    ensemble_result = calculate_metrics(weighted_logits(logits, weights), targets)
    for index, result in enumerate(individual_results, start=1):
        print(f"Seed{index}: sequence={result['sequence_acc']:.6f} "
              f"({result['correct_sequences']}/{len(targets)})")
    print(f"Ensemble: sequence={ensemble_result['sequence_acc']:.6f} "
          f"({ensemble_result['correct_sequences']}/{len(targets)})")

    print("Step 3/3: exporting ensemble errors")
    export_errors(data_root, output_dir, targets, logits, paths, weights)
    with (output_dir / "summary.txt").open("w", encoding="utf-8") as file:
        file.write(f"Validation-selected weights: {weights}\n")
        for index, result in enumerate(individual_results, start=1):
            file.write(f"Seed{index} sequence accuracy: {result['sequence_acc']:.6f}\n")
        file.write(f"Ensemble digit accuracy: {ensemble_result['digit_acc']:.6f}\n")
        file.write(f"Ensemble sequence accuracy: {ensemble_result['sequence_acc']:.6f}\n")
        file.write(f"Ensemble correct sequences: {ensemble_result['correct_sequences']}/{len(targets)}\n")
    print(f"Report saved to: {output_dir}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    main(root / "dataset_container_digits",
         [root / "releases" / "v2_9990" / "best_model.pt",
          root / "runs" / "v2_seed2" / "best_model.pt",
          root / "runs" / "v2_seed3" / "best_model.pt"],
         root / "analysis" / "v2_three_seed_ensemble")
