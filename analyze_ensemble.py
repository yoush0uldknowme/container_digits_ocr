import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from src.data import ContainerDigitsDataset, decode_digits
from src.model_v2 import Fixed6DigitCRNN
from src.model_v3 import Fixed6DigitAttentionCRNN


@dataclass
class PredictionCache:
    labels: List[str]
    paths: List[str]
    targets: torch.Tensor
    v2_logits: torch.Tensor
    v3_logits: torch.Tensor


def load_model(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


@torch.no_grad()
def collect_predictions(data_root: Path, split: str, v2_checkpoint: Path, v3_checkpoint: Path,
                        batch_size: int, num_workers: int) -> PredictionCache:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ContainerDigitsDataset(data_root, split, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    v2 = load_model(Fixed6DigitCRNN(), v2_checkpoint, device)
    v3 = load_model(Fixed6DigitAttentionCRNN(), v3_checkpoint, device)
    targets_all, v2_all, v3_all, labels_all = [], [], [], []
    for images, targets, labels in loader:
        images = images.to(device, non_blocking=True)
        targets_all.append(targets.cpu())
        v2_all.append(v2(images).float().cpu())
        v3_all.append(v3(images).float().cpu())
        labels_all.extend(labels)
    rows = [row for row in dataset.rows]
    return PredictionCache(labels_all, [row["image"] for row in rows], torch.cat(targets_all),
                           torch.cat(v2_all), torch.cat(v3_all))


def metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(targets)
    return {
        "digit_acc": correct.float().mean().item(),
        "sequence_acc": correct.all(dim=1).float().mean().item(),
        "correct_sequences": int(correct.all(dim=1).sum().item()),
    }


def choose_weight(cache: PredictionCache) -> Tuple[float, List[Dict[str, float]]]:
    results = []
    best_weight = 1.0
    best_key = (-1.0, -1.0)
    for step in range(21):
        v2_weight = step / 20.0
        combined = cache.v2_logits * v2_weight + cache.v3_logits * (1.0 - v2_weight)
        result = metrics(combined, cache.targets)
        result["v2_weight"] = v2_weight
        results.append(result)
        key = (result["sequence_acc"], result["digit_acc"])
        if key > best_key:
            best_key = key
            best_weight = v2_weight
    return best_weight, results


def export_errors(cache: PredictionCache, data_root: Path, output_dir: Path, v2_weight: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "wrong_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    v2_pred = cache.v2_logits.argmax(dim=-1)
    v3_pred = cache.v3_logits.argmax(dim=-1)
    ensemble_pred = (cache.v2_logits * v2_weight + cache.v3_logits * (1.0 - v2_weight)).argmax(dim=-1)
    rows = []
    for index, target in enumerate(cache.targets):
        true_label = decode_digits(target.tolist())
        p2 = decode_digits(v2_pred[index].tolist())
        p3 = decode_digits(v3_pred[index].tolist())
        pe = decode_digits(ensemble_pred[index].tolist())
        v2_ok, v3_ok, ensemble_ok = p2 == true_label, p3 == true_label, pe == true_label
        if not (v2_ok and v3_ok and ensemble_ok):
            source = data_root / cache.paths[index]
            name = f"{index:05d}_true-{true_label}_v2-{p2}_v3-{p3}_ens-{pe}{source.suffix}"
            shutil.copy2(source, images_dir / name)
            rows.append({"index": index, "image": cache.paths[index], "true": true_label,
                         "v2": p2, "v3": p3, "ensemble": pe,
                         "v2_correct": v2_ok, "v3_correct": v3_ok, "ensemble_correct": ensemble_ok})
    with (output_dir / "errors.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else
                                ["index", "image", "true", "v2", "v3", "ensemble", "v2_correct", "v3_correct", "ensemble_correct"])
        writer.writeheader(); writer.writerows(rows)


def main(data_root: Path, v2_checkpoint: Path, v3_checkpoint: Path, output_dir: Path,
         batch_size: int = 192, num_workers: int = 0) -> None:
    print("Collecting validation predictions...")
    validation = collect_predictions(data_root, "val", v2_checkpoint, v3_checkpoint, batch_size, num_workers)
    best_weight, weight_results = choose_weight(validation)
    print(f"Best validation weight: V2={best_weight:.2f}, V3={1.0-best_weight:.2f}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "weight_search.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["v2_weight", "digit_acc", "sequence_acc", "correct_sequences"])
        writer.writeheader(); writer.writerows(weight_results)

    print("Collecting test predictions...")
    test = collect_predictions(data_root, "test", v2_checkpoint, v3_checkpoint, batch_size, num_workers)
    v2_result = metrics(test.v2_logits, test.targets)
    v3_result = metrics(test.v3_logits, test.targets)
    ensemble_logits = test.v2_logits * best_weight + test.v3_logits * (1.0 - best_weight)
    ensemble_result = metrics(ensemble_logits, test.targets)
    print(f"V2:       sequence={v2_result['sequence_acc']:.6f} ({v2_result['correct_sequences']}/{len(test.labels)})")
    print(f"V3:       sequence={v3_result['sequence_acc']:.6f} ({v3_result['correct_sequences']}/{len(test.labels)})")
    print(f"Ensemble: sequence={ensemble_result['sequence_acc']:.6f} ({ensemble_result['correct_sequences']}/{len(test.labels)})")
    export_errors(test, data_root, output_dir, best_weight)
    with (output_dir / "summary.txt").open("w", encoding="utf-8") as file:
        file.write(f"V2 weight: {best_weight:.2f}\nV3 weight: {1.0-best_weight:.2f}\n")
        file.write(f"V2 test sequence accuracy: {v2_result['sequence_acc']:.6f}\n")
        file.write(f"V3 test sequence accuracy: {v3_result['sequence_acc']:.6f}\n")
        file.write(f"Ensemble test sequence accuracy: {ensemble_result['sequence_acc']:.6f}\n")
        file.write(f"Ensemble correct sequences: {ensemble_result['correct_sequences']}/{len(test.labels)}\n")
    print(f"Report saved to: {output_dir}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    main(root / "dataset_container_digits", root / "releases" / "v2_9990" / "best_model.pt",
         root / "runs" / "v3_attention_seed1" / "best_model.pt", root / "analysis" / "v2_v3_ensemble")

