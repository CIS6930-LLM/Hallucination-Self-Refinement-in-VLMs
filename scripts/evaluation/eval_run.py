#!/usr/bin/env python
from __future__ import annotations

import argparse

from src.evaluation.metrics import hallucination_rate, try_clip_score


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluation runner stub")
    ap.add_argument("--images", nargs="*", default=[], help="Image paths for CLIPScore")
    ap.add_argument("--texts", nargs="*", default=[], help="Texts for CLIPScore")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.images and args.texts and len(args.images) == len(args.texts):
        scores = try_clip_score(args.images, args.texts)
        if scores is None:
            print("CLIP not available — install transformers + pillow to enable.")
        else:
            print({"clip_scores": [float(s) for s in scores]})
    else:
        print("No CLIPScore inputs provided — skipping.")

    # Example hallucination rate usage
    flags = [False, True, False]
    print({"hallucination_rate": hallucination_rate(flags)})


if __name__ == "__main__":
    main()

