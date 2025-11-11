# Hallucination Self-Refinement in VLMs

Self-refinement framework that enables vision–language models to explain and verify their predictions, reducing hallucinations in captioning and VQA tasks. The project operationalizes the November 2025 research plan and provides a reproducible codebase for data curation, training, evaluation, and human studies.

## Repository Structure

```
├── configs/                 # Experiment and model configuration files
├── data/                    # Raw, processed, synthetic, and annotated datasets (placeholders only)
├── docs/                    # Research plan, architecture notes, evaluation protocols
├── experiments/             # Configs, logs, and reports for tracked runs
├── models/                  # Baseline and self-refinement checkpoints (store remotely)
├── notebooks/               # Exploratory analysis and visualization notebooks
├── scripts/                 # CLI utilities for data, training, evaluation, deployment
├── src/                     # Core Python packages (data pipeline, rationale, self-refinement, evaluation)
├── tests/                   # Automated tests mirroring src layout
└── README.md
```

Refer to the individual folder READMEs for deeper detail.

## Research Plan & Documentation

- [Updated Research Plan (Nov 2025)](docs/research_plan.md)
- [Architecture Overview](docs/architecture.md)
- [Self-Refinement VLM Overview](docs/self_refinement_overview.md)
- [Dataset Curation Workflow](docs/dataset_curation.md)
- [Human Evaluation Protocol](docs/human_evaluation_protocol.md)
- [Docs Index](docs/README.md)

## Getting Started

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. **Install core dependencies** (placeholder)
   ```bash
   pip install -r requirements.txt
   ```
   > Add the `requirements.txt` file once baseline reproduction scripts are finalized.
3. **Prepare datasets**
   - Download POPE, VQA-Hallu, and COCO into `data/raw/`.
   - Run preprocessing scripts from `scripts/data/` (TBD) to populate `data/processed/`.
4. **Configure experiments**
   - Copy or author configuration YAMLs under `configs/self_refinement/`.
   - Use `scripts/training/` entry points to launch phases P1–P6.

## Roadmap

| Phase | Timeline | Focus |
| --- | --- | --- |
| P1 | Weeks 1–3 | Data curation & synthetic triplet generation |
| P2 | Weeks 4–6 | Baseline reproduction (LLaVA, BLIP-2) |
| P3 | Weeks 7–9 | Rationale module integration |
| P4 | Weeks 10–12 | Self-check loop & contrastive training |
| P5 | Weeks 13–15 | RLAIF & human evaluation |
| P6 | Weeks 16–18 | Benchmarking, documentation, release |

## Detailed TODOs

- Environment & Tooling
  - [ ] Add `pyproject.toml`/`setup.cfg` and enable `pip install -e .` for local development.
  - [ ] Create `requirements.txt` with minimal, locked dependencies and GPU variants noted.
  - [ ] Add `pre-commit` hooks (black/isort/ruff) and a simple CI workflow (lint + unit tests).
  - [ ] Provide `.env.example` and document `PYTHONPATH` or module entry (`python -m ...`).

- Data: Ingestion & Conversion
  - [ ] Write `scripts/data/convert_vqa2_to_triplets.py` to produce JSONL with keys: `image_path`, `question`, `y1`, `rationale`, `y2` from VQA v2.
  - [ ] Implement robust answer normalization and majority-vote logic for VQA annotator answers.
  - [ ] Add dataset validator `scripts/data/validate_triplets.py` (checks missing images, empty fields, key coverage).
  - [ ] Document directory layout for raw/processed assets in `docs/dataset_curation.md` (examples, checksums, splits).
  - [ ] Optional: Add POPE/VQA-Hallu preprocessors to a common conversion interface.

- Synthetic Triplets (Optional but Recommended)
  - [ ] Build a small pipeline to generate `(y1, rationale, y2)` using a baseline VLM and prompts.
  - [ ] Add automatic sanity checks (length bounds, CLIPScore threshold) before writing to `data/processed/`.
  - [ ] Store prompts/responses for auditability under `data/synthetic/metadata/`.

- Dataset Loader & Batching
  - [ ] Extend `SelfRefineTripletDataset` to optionally load PIL images or return paths (configurable).
  - [ ] Add `torch.utils.data.DataLoader` example with `collate_triplets` and configurable batch size.
  - [ ] Add unit tests for loader, collate, and basic edge cases (empty lines, missing keys).

- VLM Adapters
  - [ ] MiniGPT‑4: Implement real model loading and `generate_*` calls in `src/self_refinement/adapters/minigpt4_adapter.py` (guarded by fallbacks).
  - [ ] LLaVA adapter scaffold (`llava_adapter.py`) with config-driven model selection and device placement.
  - [ ] BLIP‑2 adapter scaffold (`blip2_adapter.py`) with minimal generation interface.
  - [ ] Add adapter factory entries in `load_adapter` and configs for adapter-specific params.

- Self-Refine Core
  - [ ] Replace token-overlap heuristic with learned consistency scoring or multi-signal check (e.g., NLI-style, entailment classifiers, or rule-based vision cues).
  - [ ] Expose revision prompting templates via config; add few-shot exemplars option.
  - [ ] Add temperature/top‑p/top‑k generation controls and seedable sampling for reproducibility.

- Training: Supervised Fine‑Tuning (SFT)
  - [ ] Implement full training loop in `scripts/training/train_sft.py` with optimizer, LR scheduler, gradient clipping, and mixed precision.
  - [ ] Define losses for answer (`y1` and/or `y2`) and rationale; expose weights via config.
  - [ ] Add periodic evaluation on `val_jsonl` and early stopping/checkpointing under `experiments/`.
  - [ ] Integrate logging (TensorBoard/Weights & Biases) with key metrics and sample generations.

- Contrastive & Self‑Check Objectives
  - [ ] Add optional contrastive loss aligning image and rationale embeddings (InfoNCE).
  - [ ] Add rule-based or learned self‑check loss that penalizes unsupported answers.
  - [ ] Provide ablations toggled by `configs/self_refinement/*.yaml`.

- Inference & Evaluation
  - [ ] Batch inference script (`scripts/evaluation/run_inference.py`) that writes JSONL with `{initial, rationale, revised}`.
  - [ ] Metric computation script (`scripts/evaluation/compute_metrics.py`) for accuracy, hallucination rate, CLIPScore, and rationale faithfulness proxies.
  - [ ] Integrate POPE/VQA‑Hallu evaluation adapters and report templates in `experiments/reports/`.

- Experiment Management
  - [ ] Standardize config schema (data/model/train) and seed handling; include `experiments/` run folders with `config.yaml`, `metrics.json`, and `stdout.log`.
  - [ ] Update `scripts/deployment/slurm_train_sft.sh` to support adapter/model args and array jobs.
  - [ ] Provide a small `make`/`invoke` taskfile for common actions (prep, train, eval).

- Documentation
  - [ ] Expand `README.md` with quickstart (Windows/Linux), common pitfalls (PYTHONPATH), and adapter setup guides.
  - [ ] Add adapter-specific setup docs under `docs/adapters/` (MiniGPT‑4, LLaVA, BLIP‑2) with known working versions.
  - [ ] Add a tutorial notebook demonstrating end‑to‑end self‑refine on a handful of images.

- Testing & Quality
  - [ ] Unit tests for dataset, config loader, inference loop, and adapter stubs (`tests/`).
  - [ ] Golden tests for prompt templates to prevent regressions.
  - [ ] Smoke test workflow in CI to run dummy adapter over the dummy dataset.

Suggested order: start with data conversion/validation → confirm dummy training wiring → implement one real adapter (LLaVA or MiniGPT‑4) → add full SFT loop and eval → iterate on self‑check and contrastive objectives → expand docs and tests.

## Contributing

1. Fork the repository and create a feature branch.
2. Add or update tests under `tests/`.
3. Run formatting and linting (TBD).
4. Submit a pull request describing the motivation, changes, and validation.

For questions or roadmap updates, please open an issue or reach out to the team leads listed in the research plan.

## License

Distributed under the terms of the [Apache License](LICENSE).
