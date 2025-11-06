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

## Contributing

1. Fork the repository and create a feature branch.
2. Add or update tests under `tests/`.
3. Run formatting and linting (TBD).
4. Submit a pull request describing the motivation, changes, and validation.

For questions or roadmap updates, please open an issue or reach out to the team leads listed in the research plan.

## License

Distributed under the terms of the [MIT License](LICENSE).
