from pathlib import Path
from evaluate_v3 import main as evaluate_main

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "dataset_container_digits"
CHECKPOINTS = [PROJECT_ROOT / "runs" / "v3_attention_seed1" / "best_model.pt"]
# Later, add seed2/seed3 checkpoints to this list for ensemble evaluation.
SPLIT = "test"
BATCH_SIZE = 192
NUM_WORKERS = 0

if __name__ == "__main__":
    evaluate_main(data_root=DATA_ROOT, checkpoints=CHECKPOINTS, split=SPLIT,
                  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
