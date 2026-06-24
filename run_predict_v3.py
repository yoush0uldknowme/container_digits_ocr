from pathlib import Path
from predict_v3 import predict

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_ROOT / "dataset_container_digits" / "test" / "000001.jpg"
CHECKPOINTS = [PROJECT_ROOT / "runs" / "v3_attention_seed1" / "best_model.pt"]

if __name__ == "__main__":
    predict(IMAGE_PATH, CHECKPOINTS)
