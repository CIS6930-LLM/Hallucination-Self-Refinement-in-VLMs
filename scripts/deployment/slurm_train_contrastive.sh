#!/bin/bash
#SBATCH --job-name=contrastive_v1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

python -u scripts/training/train_contrastive.py --dimension 256 --batch 8

