from pathlib import Path

from evaluate import main as evaluate_main


# VS Code friendly evaluation settings.
# Change these values, then click Run on this file.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "dataset_container_digits"
CHECKPOINT = PROJECT_ROOT / "runs" / "second_fixed6" / "best_model.pt"

SPLIT = "test"
BATCH_SIZE = 256
NUM_WORKERS = 0
MAX_SAMPLES = None


def main() -> None:
    evaluate_main(
        data_root=DATA_ROOT,
        checkpoint=CHECKPOINT,
        split=SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        max_samples=MAX_SAMPLES,
    )


if __name__ == "__main__":
    main()
