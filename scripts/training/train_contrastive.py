#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from src.rationale_module.contrastive import contrastive_infonce


def parse_args():
    ap = argparse.ArgumentParser(description="Contrastive alignment training stub")
    ap.add_argument("--dimension", type=int, default=256, help="Embedding dimension")
    ap.add_argument("--batch", type=int, default=8, help="Batch size for demo")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)

    # Demo embeddings to validate loss computation.
    z_img = torch.randn(args.batch, args.dimension)
    z_txt = z_img + 0.05 * torch.randn_like(z_img)
    loss, logits = contrastive_infonce(z_img, z_txt)
    print({"loss": float(loss.item()), "logits_shape": list(logits.shape)})


if __name__ == "__main__":
    main()

