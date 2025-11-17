#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (  # type: ignore[import]
    hallucination_rate,
    try_clip_score,
)


def _normalize(text: str) -> str:
    """Simple normalization for string comparison."""
    return " ".join((text or "").strip().lower().split())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Compute basic HaloQuest metrics (accuracy, hallucination rate, "
            "optional CLIPScore) from a JSONL file."
        )
    )
    ap.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Path to JSONL file with fields including prediction and ground truth.",
    )
    ap.add_argument(
        "--pred-field",
        type=str,
        default="y2",
        help="Field name for model prediction (default: y2).",
    )
    ap.add_argument(
        "--gt-field",
        type=str,
        default="groundtruth_raw",
        help="Field name for ground-truth answer (default: groundtruth_raw).",
    )
    ap.add_argument(
        "--clip-text-field",
        type=str,
        default="y2",
        help="Field name whose text is used for CLIPScore against the image (default: y2).",
    )
    ap.add_argument(
        "--compute-clip",
        action="store_true",
        help="If set, attempt to compute CLIPScore (requires CLIP dependencies).",
    )
    return ap.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    rows = load_rows(jsonl_path)
    if not rows:
        print({"error": "no_rows"}, flush=True)
        return

    pred_field = args.pred_field
    gt_field = args.gt_field

    total = 0
    correct = 0
    halluc_flags_all: List[bool] = []
    halluc_flags_labeled: List[bool] = []

    img_paths_for_clip: List[str] = []
    texts_for_clip: List[str] = []

    for row in rows:
        pred_raw = str(row.get(pred_field, "") or "")
        gt_raw = str(row.get(gt_field, "") or "")

        if not gt_raw:
            continue

        total += 1
        pred_norm = _normalize(pred_raw)
        gt_norm = _normalize(gt_raw)
        is_correct = pred_norm == gt_norm
        if is_correct:
            correct += 1

        halluc_flags_all.append(not is_correct)

        # If the dataset tags hallucination_type, we can report a rate
        # restricted to those labeled examples as well.
        if row.get("hallucination_type"):
            halluc_flags_labeled.append(not is_correct)

        if args.compute_clip:
            image_path = row.get("image_path")
            txt = str(row.get(args.clip_text_field, "") or "")
            if image_path and txt:
                img_paths_for_clip.append(str(image_path))
                texts_for_clip.append(txt)

    accuracy = correct / total if total else 0.0
    halluc_overall = hallucination_rate(halluc_flags_all)
    halluc_labeled: Optional[float] = None
    if halluc_flags_labeled:
        halluc_labeled = hallucination_rate(halluc_flags_labeled)

    clip_mean: Optional[float] = None
    if args.compute_clip and img_paths_for_clip and texts_for_clip:
        scores = try_clip_score(img_paths_for_clip, texts_for_clip)
        if scores is not None and scores:
            clip_mean = sum(scores) / len(scores)

    summary: Dict[str, Any] = {
        "jsonl": str(jsonl_path),
        "total_examples": total,
        "accuracy": accuracy,
        "hallucination_rate_overall": halluc_overall,
        "hallucination_rate_labeled": halluc_labeled,
        "clipscore_mean": clip_mean,
        "pred_field": pred_field,
        "gt_field": gt_field,
        "clip_text_field": args.clip_text_field,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

