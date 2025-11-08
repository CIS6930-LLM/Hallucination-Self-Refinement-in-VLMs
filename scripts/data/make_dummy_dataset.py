#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def make_image(path: Path, color: tuple[int, int, int], text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (96, 96), color=color)
    d = ImageDraw.Draw(img)
    d.text((8, 40), text, fill=(255, 255, 255))
    img.save(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description="Create a small dummy dataset for self-refinement testing")
    ap.add_argument("--out", type=str, default="data/processed", help="Output folder for JSONL files")
    ap.add_argument("--imgdir", type=str, default="data/examples", help="Folder for generated images")
    ap.add_argument("--num", type=int, default=8, help="Total examples to create (>= 2)")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    img_dir = Path(args.imgdir)

    examples = []
    palette = [
        (200, 80, 80),
        (80, 200, 80),
        (80, 80, 200),
        (200, 200, 80),
        (200, 80, 200),
        (80, 200, 200),
        (140, 140, 140),
        (50, 50, 50),
    ]

    prompts = [
        ("What animal is shown?", "A cat.", "Pointy ears and whiskers suggest a cat.", "A cat."),
        ("What color is the object?", "It is red.", "Dominant red region covers the object.", "Red."),
        ("Is the person wearing glasses?", "Yes, they are.", "Frames visible around the eyes.", "Yes."),
        ("How many dogs are there?", "Two dogs.", "Two distinct dog-shaped regions are present.", "Two."),
        ("Is it daytime or nighttime?", "Daytime.", "Bright sky and shadows indicate daylight.", "Daytime."),
        ("What is the weather like?", "It is sunny.", "Clear sky and bright lighting suggest sun.", "Sunny."),
        ("What is on the table?", "A laptop.", "Rectangular device with keyboard visible.", "A laptop."),
        ("Is the car moving?", "No, it is parked.", "No motion blur and wheels aligned at curb.", "Parked."),
    ]

    n = max(2, min(args.num, len(prompts)))
    for i in range(n):
        img_path = img_dir / f"img_{i}.png"
        make_image(img_path, palette[i % len(palette)], f"{i}")
        q, y1, r, y2 = prompts[i]
        examples.append({
            "image_path": str(img_path).replace("\\", "/"),
            "question": q,
            "y1": y1,
            "rationale": r,
            "y2": y2,
        })

    # Split 70/30 into train/val
    k = max(1, int(0.7 * len(examples)))
    train_rows = examples[:k]
    val_rows = examples[k:]
    write_jsonl(out_dir / "self_refine_triplets_train.jsonl", train_rows)
    write_jsonl(out_dir / "self_refine_triplets_val.jsonl", val_rows)

    print({
        "train_path": str(out_dir / "self_refine_triplets_train.jsonl"),
        "val_path": str(out_dir / "self_refine_triplets_val.jsonl"),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "images_dir": str(img_dir),
    })


if __name__ == "__main__":
    main()

