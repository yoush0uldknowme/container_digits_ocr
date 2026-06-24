from pathlib import Path
from evaluate_v2_three_seed import main

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "dataset_container_digits"
CHECKPOINTS = [
    PROJECT_ROOT / "releases" / "v2_9990" / "best_model.pt",
    PROJECT_ROOT / "runs" / "v2_seed2" / "best_model.pt",
    PROJECT_ROOT / "runs" / "v2_seed3" / "best_model.pt",
]
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "v2_three_seed_ensemble"
BATCH_SIZE = 192
NUM_WORKERS = 0

if __name__ == "__main__":
    main(DATA_ROOT, CHECKPOINTS, OUTPUT_DIR, BATCH_SIZE, NUM_WORKERS)
