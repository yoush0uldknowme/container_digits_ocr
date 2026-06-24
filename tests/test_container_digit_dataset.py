import csv
import re
import tempfile
import unittest
from pathlib import Path

from generate_container_digit_dataset import DatasetConfig, build_split_plan, generate_dataset


class ContainerDigitDatasetTests(unittest.TestCase):
    def test_split_plan_preserves_requested_counts(self):
        config = DatasetConfig(total=100000, train=80000, val=10000, test=10000)

        split_plan = build_split_plan(config)

        self.assertEqual(len(split_plan), 100000)
        self.assertEqual(split_plan.count("train"), 80000)
        self.assertEqual(split_plan.count("val"), 10000)
        self.assertEqual(split_plan.count("test"), 10000)

    def test_small_generation_writes_images_labels_and_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "dataset"
            config = DatasetConfig(total=12, train=8, val=2, test=2, seed=7)

            generate_dataset(config, output_dir)

            with (output_dir / "labels.csv").open(newline="", encoding="utf-8") as labels_file:
                rows = list(csv.DictReader(labels_file))
            self.assertEqual(len(rows), 12)
            self.assertTrue((output_dir / "config.json").exists())
            self.assertEqual(len(list((output_dir / "train").glob("*.jpg"))), 8)
            self.assertEqual(len(list((output_dir / "val").glob("*.jpg"))), 2)
            self.assertEqual(len(list((output_dir / "test").glob("*.jpg"))), 2)
            self.assertGreaterEqual(len(list((output_dir / "preview").glob("*.jpg"))), 1)

            for row in rows:
                self.assertRegex(row["label"], re.compile(r"^\d{6}$"))
                self.assertIn(row["split"], {"train", "val", "test"})
                self.assertTrue((output_dir / row["image"]).exists())


if __name__ == "__main__":
    unittest.main()
