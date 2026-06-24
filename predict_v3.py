from pathlib import Path
from typing import List

import torch
from PIL import Image

from evaluate_v3 import load_models
from src.data import build_transform, decode_digits


def predict(image: Path, checkpoints: List[Path]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(checkpoints, device)
    tensor = build_transform(train=False)(Image.open(image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad(): logits = torch.stack([model(tensor) for model in models]).mean(dim=0)
    probabilities = logits.softmax(dim=-1); digits = probabilities.argmax(dim=-1).squeeze(0)
    confidence = probabilities.max(dim=-1).values.squeeze(0)
    print(f"prediction: {decode_digits(digits.cpu().tolist())}")
    print("confidence: " + " ".join(f"{value*100:.2f}%" for value in confidence.cpu().tolist()))


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    predict(ROOT / "dataset_container_digits" / "test" / "000001.jpg", [ROOT / "runs" / "v3_attention_seed1" / "best_model.pt"])
