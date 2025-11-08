#!/bin/bash
#SBATCH --job-name=sft_v1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

CONFIG=${CONFIG:-configs/self_refinement/default.yaml}
TRAIN_JSONL=${TRAIN_JSONL:-data/processed/self_refine_triplets_train.jsonl}
ADAPTER=${ADAPTER:-dummy}

echo "Running SFT with $CONFIG"
python -u scripts/training/train_sft.py \
  --config "$CONFIG" \
  --adapter "$ADAPTER" \
  --train_jsonl "$TRAIN_JSONL"

