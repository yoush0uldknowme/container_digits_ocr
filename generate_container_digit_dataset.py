import argparse
import csv
import io
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: Pillow. Install it with `python -m pip install -r requirements.txt`."
    ) from exc


@dataclass(frozen=True)
class DatasetConfig:
    total: int = 100000
    train: int = 80000
    val: int = 10000
    test: int = 10000
    width: int = 192
    height: int = 64
    seed: int = 20260527
    image_format: str = "jpg"
    preview_count: int = 60


BACKGROUND_PALETTE = (
    (174, 178, 170),
    (132, 48, 42),
    (73, 95, 112),
    (116, 98, 74),
    (204, 199, 185),
    (46, 73, 88),
)

TEXT_PALETTE = (
    (18, 18, 18),
    (230, 226, 212),
    (238, 232, 190),
    (30, 35, 40),
    (205, 210, 205),
)

BLOCKED_FONT_NAME_TOKENS = (
    "webding",
    "wingding",
    "symbol",
    "marlett",
    "segmdl2",
    "holomdl2",
)


def filter_usable_fonts(fonts: list[Path]) -> list[Path]:
    usable = []
    for path in fonts:
        name = path.stem.lower()
        if any(token in name for token in BLOCKED_FONT_NAME_TOKENS):
            continue
        usable.append(path)
    return usable


def build_split_plan(config: DatasetConfig) -> list[str]:
    if config.total != config.train + config.val + config.test:
        raise ValueError("total must equal train + val + test")
    return ["train"] * config.train + ["val"] * config.val + ["test"] * config.test


def find_fonts() -> list[Path]:
    candidates = []
    roots = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts"),
        Path("/Library/Fonts"),
    ]
    preferred = (
        "arial",
        "bahnschrift",
        "calibri",
        "consola",
        "cour",
        "impact",
        "simhei",
        "msyh",
        "din",
    )
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".ttf", ".otf", ".ttc"}:
                continue
            name = path.stem.lower()
            if any(token in name for token in preferred):
                candidates.append(path)
    return filter_usable_fonts(candidates)[:80]


def random_label(rng: random.Random) -> str:
    return f"{rng.randrange(0, 1000000):06d}"


def load_font(fonts: list[Path], rng: random.Random, size: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    if fonts:
        return ImageFont.truetype(str(rng.choice(fonts)), size=size)
    return ImageFont.load_default()


def add_background_texture(image: Image.Image, rng: random.Random) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    for _ in range(rng.randint(60, 130)):
        x = rng.randint(0, width - 1)
        y = rng.randint(0, height - 1)
        length = rng.randint(4, 35)
        shade = rng.randint(-35, 35)
        color = (max(0, min(255, 120 + shade)),) * 3 + (rng.randint(12, 45),)
        draw.line((x, y, min(width, x + length), y + rng.randint(-2, 2)), fill=color, width=1)
    for _ in range(rng.randint(2, 7)):
        x0 = rng.randint(-15, width - 10)
        y0 = rng.randint(-10, height - 8)
        x1 = x0 + rng.randint(10, 55)
        y1 = y0 + rng.randint(5, 28)
        rust = (rng.randint(95, 160), rng.randint(45, 85), rng.randint(25, 55), rng.randint(18, 55))
        draw.ellipse((x0, y0, x1, y1), fill=rust)


def draw_digits(label: str, config: DatasetConfig, rng: random.Random, fonts: list[Path]) -> Image.Image:
    layer = Image.new("RGBA", (config.width, config.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font_size = rng.randint(int(config.height * 0.55), int(config.height * 0.82))
    font = load_font(fonts, rng, font_size)
    spacing = rng.randint(-1, 4)
    widths = [draw.textbbox((0, 0), char, font=font)[2] for char in label]
    total_width = sum(widths) + spacing * (len(label) - 1)
    x = max(2, (config.width - total_width) // 2 + rng.randint(-10, 10))
    y = rng.randint(1, max(2, config.height - font_size - 2))
    text_color = rng.choice(TEXT_PALETTE) + (rng.randint(190, 255),)

    for char, char_width in zip(label, widths):
        jitter_y = y + rng.randint(-2, 2)
        if rng.random() < 0.45:
            shadow = (0, 0, 0, rng.randint(35, 80))
            draw.text((x + rng.randint(1, 2), jitter_y + rng.randint(1, 2)), char, font=font, fill=shadow)
        draw.text((x, jitter_y), char, font=font, fill=text_color)
        if rng.random() < 0.22:
            draw.text((x + 1, jitter_y), char, font=font, fill=text_color)
        x += char_width + spacing

    angle = rng.uniform(-4.0, 4.0)
    layer = layer.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    if rng.random() < 0.35:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 0.75)))
    return layer


def add_occlusions(image: Image.Image, rng: random.Random) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    if rng.random() < 0.35:
        for _ in range(rng.randint(1, 3)):
            y = rng.randint(5, height - 8)
            x0 = rng.randint(0, max(0, width - 24))
            x1 = min(width, x0 + rng.randint(20, max(21, width // 2)))
            draw.rectangle(
                (x0, y, x1, y + rng.randint(1, 4)),
                fill=(rng.randint(60, 150), rng.randint(45, 110), rng.randint(30, 90), rng.randint(35, 85)),
            )
    if rng.random() < 0.25:
        for _ in range(rng.randint(1, 4)):
            x = rng.randint(0, width)
            draw.line((x, 0, x + rng.randint(-10, 10), height), fill=(255, 255, 255, rng.randint(12, 32)), width=1)


def jpeg_roundtrip(image: Image.Image, rng: random.Random) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(55, 92), optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def generate_image(label: str, config: DatasetConfig, rng: random.Random, fonts: list[Path]) -> Image.Image:
    base_color = rng.choice(BACKGROUND_PALETTE)
    image = Image.new("RGB", (config.width, config.height), base_color)
    add_background_texture(image, rng)
    image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.35)))

    digits = draw_digits(label, config, rng, fonts)
    image = Image.alpha_composite(image.convert("RGBA"), digits)
    add_occlusions(image, rng)
    image = image.convert("RGB")

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.72, 1.25))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.45))
    if rng.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.65)))
    return jpeg_roundtrip(image, rng)


def save_image(image: Image.Image, path: Path, image_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_format.lower() in {"jpg", "jpeg"}:
        image.save(path, format="JPEG", quality=88)
    elif image_format.lower() == "png":
        image.save(path, format="PNG")
    else:
        raise ValueError("image_format must be jpg or png")


def generate_dataset(config: DatasetConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    fonts = find_fonts()
    split_plan = build_split_plan(config)
    extension = "jpg" if config.image_format.lower() in {"jpg", "jpeg"} else "png"

    with (output_dir / "labels.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image", "label", "split"])
        writer.writeheader()
        for index, split in enumerate(split_plan, start=1):
            label = random_label(rng)
            image = generate_image(label, config, rng, fonts)
            split_index = index if split == "train" else index - config.train if split == "val" else index - config.train - config.val
            relative_path = Path(split) / f"{split_index:06d}.{extension}"
            save_image(image, output_dir / relative_path, config.image_format)
            writer.writerow({"image": relative_path.as_posix(), "label": label, "split": split})

            if index <= config.preview_count:
                preview_path = output_dir / "preview" / f"{index:03d}_{label}.{extension}"
                save_image(image, preview_path, config.image_format)

    metadata = asdict(config)
    metadata["fonts_found"] = [str(path) for path in fonts]
    (output_dir / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic 6-digit container-number crop images.")
    parser.add_argument("--output", type=Path, default=Path("dataset_container_digits"))
    parser.add_argument("--total", type=int, default=100000)
    parser.add_argument("--train", type=int, default=80000)
    parser.add_argument("--val", type=int, default=10000)
    parser.add_argument("--test", type=int, default=10000)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--preview-count", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetConfig(
        total=args.total,
        train=args.train,
        val=args.val,
        test=args.test,
        width=args.width,
        height=args.height,
        seed=args.seed,
        image_format=args.format,
        preview_count=args.preview_count,
    )
    generate_dataset(config, args.output)
    print(f"Generated {config.total} images at {args.output}")


if __name__ == "__main__":
    main()

