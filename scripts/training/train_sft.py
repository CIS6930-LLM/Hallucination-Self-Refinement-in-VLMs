#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from src.data_pipeline.datasets import SelfRefineTripletDataset
from src.self_refinement.self_refine_vlm import SelfRefineVLM, SelfRefineConfig, load_adapter
from src.utils.config import load_yaml_config


def parse_args():
    ap = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) for self-refinement")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--adapter", type=str, default="dummy", help="Base VLM adapter name (e.g., dummy, llava)")
    ap.add_argument("--train_jsonl", type=str, required=True, help="Path to triplet JSONL")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    ds = SelfRefineTripletDataset(args.train_jsonl)

    adapter = load_adapter(args.adapter)
    model = SelfRefineVLM(adapter=adapter, config=SelfRefineConfig())

    # Minimal demonstration: run one pass to validate wiring.
    # Replace with actual tokenization and model fine-tuning when integrating a real adapter.
    if len(ds) == 0:
        print("Dataset is empty — nothing to train.")
        return

    sample = ds[0]
    image_path = sample["image_path"]
    question = sample["question"]
    y1 = model.generate_answer(image_path, question)
    r = model.generate_rationale(image_path, question, y1)
    y2 = model.revise_answer(image_path, question, y1, r)

    print("[SFT] Wiring check successful.")
    print({"initial": y1, "rationale": r, "revised": y2})


if __name__ == "__main__":
    main()

