import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from PIL import Image

from src.data import build_transform, decode_digits
from src.model_v2 import Fixed6DigitCRNN


DEFAULT_WEIGHTS = (0.3, 0.7)
DEFAULT_THRESHOLD = 0.99


def load_models(checkpoints: Sequence[Path], device: torch.device) -> List[Fixed6DigitCRNN]:
    models = []
    for path in checkpoints:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = Fixed6DigitCRNN().to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)
    return models


def load_threshold(config_path: Path, fallback: float = DEFAULT_THRESHOLD) -> float:
    if not config_path.exists():
        return fallback
    return float(json.loads(config_path.read_text(encoding="utf-8"))["threshold"])


@torch.no_grad()
def predict_reliable(image_path: Path, checkpoints: Sequence[Path], weights: Sequence[float],
                     threshold: float) -> Dict[str, object]:
    if len(checkpoints) != len(weights):
        raise ValueError("checkpoints and weights must have the same length")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(checkpoints, device)
    image = Image.open(image_path).convert("RGB")
    tensor = build_transform(train=False)(image).unsqueeze(0).to(device)
    logits = sum(model(tensor) * weight for model, weight in zip(models, weights))
    probabilities = logits.softmax(dim=-1).squeeze(0)
    digits = probabilities.argmax(dim=-1)
    digit_confidences = probabilities.max(dim=-1).values
    min_confidence = float(digit_confidences.min().item())
    mean_confidence = float(digit_confidences.mean().item())
    return {
        "prediction": decode_digits(digits.cpu().tolist()),
        "digit_confidences": [float(value) for value in digit_confidences.cpu().tolist()],
        "min_digit_confidence": min_confidence,
        "mean_digit_confidence": mean_confidence,
        "threshold": threshold,
        "status": "ACCEPTED" if min_confidence >= threshold else "MANUAL_REVIEW",
    }


def print_result(result: Dict[str, object]) -> None:
    print(f"prediction: {result['prediction']}")
    print("digit confidence: " + " ".join(f"{value * 100:.2f}%" for value in result["digit_confidences"]))
    print(f"minimum confidence: {result['min_digit_confidence'] * 100:.2f}%")
    print(f"threshold: {result['threshold'] * 100:.2f}%")
    print(f"status: {result['status']}")
