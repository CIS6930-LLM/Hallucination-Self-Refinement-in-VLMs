# Dataset Curation & Synthetic Generation

## Source Datasets

- **POPE Benchmark:** Object hallucination detection pairs; ingest via official repository.
- **VQA-Hallu:** Question-answer pairs with hallucination labels; request academic access if needed.
- **COCO Captions:** Base corpus for captioning; combine with synthetic distractors for grounding stress tests.

## Preprocessing Workflow

1. **Acquisition:** Download raw archives to `data/raw/` and verify checksums.  
2. **Normalization:** Standardize file naming, convert annotations to JSONL, and normalize bounding boxes.  
3. **Filtering:** Remove corrupted images, flag ambiguous cases for manual review, and map label taxonomies across datasets.  
4. **Splitting:** Produce train/validation/test splits with consistent seed controls; store in `data/processed/`.  
5. **Metadata Tracking:** Log dataset version, preprocessing scripts, and commit hash in `experiments/reports/dataset_changelog.md`.

## Synthetic Triplet Generation

- Use GPT-4V (or comparable VLM) to produce `(y₁, r, y₂)` triplets.  
- Prompt template should elicit initial response, rationale extraction, and self-refined revision.  
- Store intermediate prompts/responses to enable auditability.  
- Apply automatic sanity checks (CLIPScore threshold, length constraints) before promotion to `data/synthetic/`.

## Data Quality Assurance

- Run hallucination detectors (baseline VLMs) to estimate noise.  
- Sample 5% of synthetic data for human spot-checks.  
- Track inter-run drift with summary statistics (token length, rationale coverage, label distribution).

## Privacy & Licensing

- Respect dataset licenses (COCO Creative Commons, POPE/VQA academic use).  
- Exclude personally identifiable information.  
- Document any additional sources or derived datasets.
