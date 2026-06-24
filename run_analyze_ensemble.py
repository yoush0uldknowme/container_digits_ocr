from pathlib import Path
from analyze_ensemble import main

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "dataset_container_digits"
V2_CHECKPOINT = PROJECT_ROOT / "releases" / "v2_9990" / "best_model.pt"
V3_CHECKPOINT = PROJECT_ROOT / "runs" / "v3_attention_seed1" / "best_model.pt"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "v2_v3_ensemble"
BATCH_SIZE = 192
NUM_WORKERS = 0

if __name__ == "__main__":
    main(DATA_ROOT, V2_CHECKPOINT, V3_CHECKPOINT, OUTPUT_DIR, BATCH_SIZE, NUM_WORKERS)
