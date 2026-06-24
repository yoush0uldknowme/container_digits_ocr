import argparse
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from src.data import build_transform, decode_digits
from src.model_v2 import Fixed6DigitCRNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one 6-digit container number crop with v2 CNN+BiLSTM.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/v2_bilstm/best_model.pt"))
    return parser.parse_args()


def main(image: Optional[Path] = None, checkpoint: Optional[Path] = None) -> None:
    if image is None and checkpoint is None:
        args = parse_args()
    else:
        args = argparse.Namespace(
            image=image,
            checkpoint=checkpoint or Path("runs/v2_bilstm/best_model.pt"),
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Fixed6DigitCRNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = build_transform(train=False)
    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    prediction = decode_digits(logits.argmax(dim=-1).squeeze(0).cpu().tolist())
    print(f"prediction: {prediction}")


if __name__ == "__main__":
    main()

