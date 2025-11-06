# Data Directory

This folder hosts all datasets required for training, evaluation, and synthetic generation.

- `raw/`: Original datasets such as POPE, VQA-Hallu, and COCO captions prior to preprocessing.
- `processed/`: Cleaned and feature-aligned splits ready for training or evaluation.
- `synthetic/`: Triplets `(y1, r, y2)` generated via large multimodal models or augmentation pipelines.
- `annotations/`: Human evaluation outputs and rationale annotations.

Keep large binaries out of version control; use the provided `.gitkeep` placeholders and reference data storage in documentation.
