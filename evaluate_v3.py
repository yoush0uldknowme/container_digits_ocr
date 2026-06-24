import argparse
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from src.data import ContainerDigitsDataset
from src.model_v3 import Fixed6DigitAttentionCRNN


def load_models(checkpoints: List[Path], device: torch.device) -> List[Fixed6DigitAttentionCRNN]:
    models = []
    for path in checkpoints:
        model = Fixed6DigitAttentionCRNN().to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"]); model.eval(); models.append(model)
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one or more V3 checkpoints by probability averaging.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset_container_digits"))
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=[Path("runs/v3_attention_seed1/best_model.pt")])
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main(data_root: Optional[Path] = None, checkpoints: Optional[List[Path]] = None, split: Optional[str] = None,
         batch_size: Optional[int] = None, num_workers: Optional[int] = None) -> None:
    args = parse_args()
    if data_root is not None: args.data_root = data_root
    if checkpoints is not None: args.checkpoints = checkpoints
    if split is not None: args.split = split
    if batch_size is not None: args.batch_size = batch_size
    if num_workers is not None: args.num_workers = num_workers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ContainerDigitsDataset(args.data_root, args.split, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    models = load_models(args.checkpoints, device)
    total_loss = 0.0; total_sequences = 0; correct_digits = 0; correct_sequences = 0
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device); targets = targets.to(device)
            logits = torch.stack([model(images) for model in models]).mean(dim=0)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
            predictions = logits.argmax(dim=-1); correct = predictions.eq(targets)
            total_loss += loss.item() * images.size(0); total_sequences += images.size(0)
            correct_digits += correct.sum().item(); correct_sequences += correct.all(dim=1).sum().item()
    print(f"split: {args.split}"); print(f"models: {len(models)}")
    print(f"loss: {total_loss/total_sequences:.6f}")
    print(f"digit_acc: {correct_digits/(total_sequences*6):.6f}")
    print(f"sequence_acc: {correct_sequences/total_sequences:.6f}")
    print(f"correct_sequences: {correct_sequences}/{total_sequences}")


if __name__ == "__main__": main()
