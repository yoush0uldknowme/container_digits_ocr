import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from generate_container_digit_dataset import filter_usable_fonts
from src.data import ContainerDigitsDataset, decode_digits, encode_label
from src.metrics import compute_digit_and_sequence_accuracy
from src.model import Fixed6DigitCNN


def write_tiny_dataset(root: Path) -> None:
    rows = [
        ("train/000001.jpg", "031106", "train", (80, 90, 100)),
        ("val/000001.jpg", "987650", "val", (120, 70, 60)),
        ("test/000001.jpg", "000123", "test", (30, 40, 50)),
    ]
    for image_path, _label, _split, color in rows:
        path = root / image_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (192, 64), color).save(path)
    with (root / "labels.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image", "label", "split"])
        for image_path, label, split, _color in rows:
            writer.writerow([image_path, label, split])


class Fixed6TrainingTests(unittest.TestCase):
    def test_label_encode_decode_preserves_leading_zero(self):
        encoded = encode_label("031106")

        self.assertEqual(encoded, [0, 3, 1, 1, 0, 6])
        self.assertEqual(decode_digits(encoded), "031106")

    def test_dataset_loads_requested_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_tiny_dataset(root)

            dataset = ContainerDigitsDataset(root, split="train", augment=False)
            image, target, label = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(tuple(image.shape), (3, 64, 192))
            self.assertEqual(target.tolist(), [0, 3, 1, 1, 0, 6])
            self.assertEqual(label, "031106")

    def test_model_outputs_six_digit_logits(self):
        model = Fixed6DigitCNN()
        batch = torch.zeros(2, 3, 64, 192)

        logits = model(batch)

        self.assertEqual(tuple(logits.shape), (2, 6, 10))

    def test_font_filter_rejects_symbol_fonts(self):
        fonts = [
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/webdings.ttf"),
            Path("C:/Windows/Fonts/wingding.ttf"),
        ]

        usable = filter_usable_fonts(fonts)

        self.assertIn(Path("C:/Windows/Fonts/arial.ttf"), usable)
        self.assertNotIn(Path("C:/Windows/Fonts/webdings.ttf"), usable)
        self.assertNotIn(Path("C:/Windows/Fonts/wingding.ttf"), usable)

    def test_model_uses_positionwise_classifier(self):
        model = Fixed6DigitCNN()

        self.assertTrue(hasattr(model, "position_classifier"))

    def test_metrics_report_digit_and_sequence_accuracy(self):
        logits = torch.zeros(2, 6, 10)
        targets = torch.tensor([[0, 1, 2, 3, 4, 5], [9, 8, 7, 6, 5, 4]])
        for batch_index, digits in enumerate(targets.tolist()):
            for pos, digit in enumerate(digits):
                logits[batch_index, pos, digit] = 10.0
        logits[1, 5] = 0.0
        logits[1, 5, 0] = 10.0

        digit_acc, sequence_acc = compute_digit_and_sequence_accuracy(logits, targets)

        self.assertAlmostEqual(digit_acc, 11 / 12)
        self.assertAlmostEqual(sequence_acc, 0.5)


if __name__ == "__main__":
    unittest.main()
