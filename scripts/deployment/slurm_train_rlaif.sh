#!/bin/bash
#SBATCH --job-name=rlaif_v1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

python -u scripts/training/train_rlaif.py --note rlaif_scaffold

