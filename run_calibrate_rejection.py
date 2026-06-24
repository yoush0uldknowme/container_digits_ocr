from pathlib import Path
from calibrate_rejection import main

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "dataset_container_digits"
CHECKPOINTS = [
    PROJECT_ROOT / "runs" / "v2_seed2" / "best_model.pt",
    PROJECT_ROOT / "runs" / "v2_seed3" / "best_model.pt",
]
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "reliable_inference"
TARGET_ACCEPTED_ACCURACY = 0.9999
BATCH_SIZE = 192
NUM_WORKERS = 0

if __name__ == "__main__":
    main(DATA_ROOT, CHECKPOINTS, OUTPUT_DIR, TARGET_ACCEPTED_ACCURACY, BATCH_SIZE, NUM_WORKERS)
