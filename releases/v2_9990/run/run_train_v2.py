from pathlib import Path

from train_v2 import main as train_main


# VS Code friendly v2 training settings.
# Change these values, then click Run on this file.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "dataset_container_digits"
RUN_DIR = PROJECT_ROOT / "runs" / "v2_bilstm_testtime"

EPOCHS = 60
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

# Use small values for a quick check, or None for full training.
TRAIN_MAX_SAMPLES = None
VAL_MAX_SAMPLES = None


def main() -> None:
    train_main(
        data_root=DATA_ROOT,
        run_dir=RUN_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        num_workers=NUM_WORKERS,
        train_max_samples=TRAIN_MAX_SAMPLES,
        val_max_samples=VAL_MAX_SAMPLES,
    )


if __name__ == "__main__":
    main()
