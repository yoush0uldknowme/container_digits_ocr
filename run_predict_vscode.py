from pathlib import Path

from predict import main as predict_main


# VS Code friendly single-image prediction settings.
# Change IMAGE_PATH, then click Run on this file.
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT = PROJECT_ROOT / "runs" / "first_fixed6" / "best_model.pt"
IMAGE_PATH = PROJECT_ROOT / "dataset_container_digits" / "test" / "000001.jpg"


def main() -> None:
    predict_main(image=IMAGE_PATH, checkpoint=CHECKPOINT)


if __name__ == "__main__":
    main()
