from pathlib import Path

from reliable_inference import DEFAULT_WEIGHTS, load_threshold, predict_reliable, print_result

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_ROOT / "dataset_container_digits" / "test" / "000002.jpg"
CHECKPOINTS = [
    PROJECT_ROOT / "runs" / "v2_seed2" / "best_model.pt",
    PROJECT_ROOT / "runs" / "v2_seed3" / "best_model.pt",
]
WEIGHTS = DEFAULT_WEIGHTS
THRESHOLD_CONFIG = PROJECT_ROOT / "analysis" / "reliable_inference" / "threshold.json"

if __name__ == "__main__":
    threshold = load_threshold(THRESHOLD_CONFIG)
    result = predict_reliable(IMAGE_PATH, CHECKPOINTS, WEIGHTS, threshold)
    print_result(result)
