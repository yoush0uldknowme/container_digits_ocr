from pathlib import Path
from train_v3 import main as train_main

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "dataset_container_digits"
RUN_DIR = PROJECT_ROOT / "runs" / "v3_attention_seed1"
EPOCHS = 80
BATCH_SIZE = 96
LEARNING_RATE = 7e-4
NUM_WORKERS = 0
SEED = 20260611
TRAIN_MAX_SAMPLES = None
VAL_MAX_SAMPLES = None

if __name__ == "__main__":
    train_main(data_root=DATA_ROOT, run_dir=RUN_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE,
               lr=LEARNING_RATE, num_workers=NUM_WORKERS, seed=SEED,
               train_max_samples=TRAIN_MAX_SAMPLES, val_max_samples=VAL_MAX_SAMPLES)
