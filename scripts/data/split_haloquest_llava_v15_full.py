#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Split haloquest_llava_v15_full_with_rationales.jsonl into train/val/test."
    )
    ap.add_argument(
        "--in_jsonl",
        type=str,
        default=(
            "/blue/cis6930/jatin.salve/"
            "Hallucination-Self-Refinement-in-VLMs/"
            "data/processed/haloquest_llava_v15_full_with_rationales.jsonl"
        ),
        help="Input JSONL with image_path, question, y1, rationale, y2/groundtruth_raw.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=(
            "/blue/cis6930/jatin.salve/"
            "Hallucination-Self-Refinement-in-VLMs/data/processed"
        ),
        help="Output directory for split JSONLs.",
    )
    ap.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of examples for train split.",
    )
    ap.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of examples for validation split.",
    )
    ap.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of examples for test split.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before splitting.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_jsonl)
    out_dir = Path(args.out_dir)

    if not in_path.exists():
        raise FileNotFoundError(in_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all rows.
    rows: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No valid rows found in {in_path}")

    n = len(rows)
    random.Random(args.seed).shuffle(rows)

    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val :]

    def write_split(split_name: str, split_rows: List[Dict[str, Any]]) -> None:
        out_path = out_dir / f"haloquest_llava_v15_full_with_rationales_{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in split_rows:
                # Tag with new split label (in addition to any existing one).
                r["split_lora"] = split_name
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_split("train", train_rows)
    write_split("val", val_rows)
    write_split("test", test_rows)

    print(
        {
            "in": str(in_path),
            "total": n,
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "n_test": len(test_rows),
            "out_dir": str(out_dir),
        }
    )


if __name__ == "__main__":
    main()

