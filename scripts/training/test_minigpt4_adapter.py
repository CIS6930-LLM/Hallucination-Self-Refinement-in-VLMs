#!/usr/bin/env python
from __future__ import annotations

import argparse

from src.self_refinement.adapters.minigpt4_adapter import MiniGPT4Adapter, MiniGPT4Init
from src.self_refinement.self_refine_vlm import SelfRefineVLM
from src.self_refinement.inference import self_refine_inference


def parse_args():
    ap = argparse.ArgumentParser(description="Test MiniGPT-4 adapter scaffold")
    ap.add_argument("--repo", type=str, default="", help="Path to local MiniGPT-4 repo (optional)")
    ap.add_argument("--config_yaml", type=str, default="", help="MiniGPT-4 config YAML (optional)")
    ap.add_argument("--ckpt", type=str, default="", help="Checkpoint path (optional)")
    ap.add_argument("--device", type=str, default="", help="Device, e.g., cuda or cpu")
    ap.add_argument("--image", type=str, default="data/examples/img_0.png", help="Test image path")
    ap.add_argument("--question", type=str, default="What animal is shown?", help="Test question")
    ap.add_argument("--no-fallback", action="store_true", help="Disable stub fallback if MiniGPT-4 is missing")
    return ap.parse_args()


def main():
    args = parse_args()
    init = MiniGPT4Init(
        repo_path=args.repo or None,
        config_yaml=args.config_yaml or None,
        ckpt_path=args.ckpt or None,
        device=args.device or None,
        fallback_on_missing=not args.no_fallback,
    )
    adapter = MiniGPT4Adapter(init)
    model = SelfRefineVLM(adapter)
    print({"adapter_info": adapter.info})
    res = self_refine_inference(model, args.image, args.question)
    print(res)


if __name__ == "__main__":
    main()

