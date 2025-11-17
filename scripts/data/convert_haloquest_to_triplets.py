#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert HaloQuest CSV to self-refinement triplet JSONL."
    )
    ap.add_argument(
        "--csv",
        type=str,
        default="/blue/cis6930/jatin.salve/Hallucination-Self-Refinement-in-VLMs/data/excelsheet/haloquest-train.csv",
        help="Path to haloquest-train.csv",
    )
    ap.add_argument(
        "--imgdir",
        type=str,
        default="/blue/cis6930/jatin.salve/image/downloads",
        help="Directory containing downloaded images.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="/blue/cis6930/jatin.salve/Hallucination-Self-Refinement-in-VLMs/data/processed/haloquest_triplets.jsonl",
        help="Output JSONL path for triplets.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    img_dir = Path(args.imgdir)
    out_path = Path(args.out)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not img_dir.exists():
        raise FileNotFoundError(img_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped_missing_img = 0
    skipped_no_name = 0

    with csv_path.open("r", encoding="utf-8") as f_in, out_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            image_name = (row.get("image_name") or "").strip()
            if not image_name:
                skipped_no_name += 1
                continue

            img_path = img_dir / image_name
            if not img_path.exists():
                skipped_missing_img += 1
                continue

            question = (row.get("question") or "").strip()
            gt = (row.get("groundtruth responses") or "").strip()

            # Map HaloQuest fields to self-refinement triplet schema.
            # We leave y1 and rationale empty for now so they can be
            # filled by a model or later processing; y2 stores the
            # ground-truth response string (possibly with ';'-separated variants).
            example = {
                # Store a path relative to the project root so it works
                # with SelfRefineTripletDataset without needing image_root.
                "image_path": str(img_path),
                "question": question,
                "y1": "",
                "rationale": "",
                "y2": gt,
                # Optional metadata for downstream analysis
                "hallucination_type": row.get("hallucination type", "").strip(),
                "image_type": row.get("image type", "").strip(),
                "split": row.get("split", "").strip(),
            }
            f_out.write(json.dumps(example, ensure_ascii=False) + "\n")
            kept += 1

    print(
        {
            "csv": str(csv_path),
            "imgdir": str(img_dir),
            "out": str(out_path),
            "kept": kept,
            "skipped_missing_img": skipped_missing_img,
            "skipped_no_name": skipped_no_name,
        }
    )


if __name__ == "__main__":
    main()
