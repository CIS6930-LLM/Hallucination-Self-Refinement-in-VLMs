# Human Evaluation Protocol

## Overview

This document captures the annotation rubric and workflow used by the graduate student panel and faculty supervisor. It ensures consistency with the research plan and provides traceability for each assessed sample.

## Roles & Responsibilities

- **Evaluation Lead:** Coordinates sampling, assigns tasks, and aggregates results.
- **Annotators (3x graduate students):** Score factuality, relevance, and confidence per instance.
- **Faculty Supervisor:** Performs weekly audits and resolves disagreements.

## Sampling Strategy

- 300 captioning outputs and 200 VQA responses per evaluation cycle.
- Stratify by dataset (POPE, VQA-Hallu, synthetic corpus) and model variant (`baseline`, `self-refinement`, `ablations`).
- Randomize order within the annotation interface to limit bias.

## Scoring Rubric (0–5 Scale)

- **Factuality:** Degree to which textual claims align with visual evidence.
- **Relevance:** Alignment of the response with the prompt or question.
- **Confidence:** Annotator’s certainty in their judgement.

Use whole numbers only; reserve `0` for unusable or off-topic outputs.

## Agreement Targets

- Minimum pairwise Cohen’s κ of 0.75 across annotators.
- Trigger adjudication when κ < 0.6 or when standard deviation exceeds 1.5 points.

## Tooling & Logging

- Annotation interface: web-based dashboard (TBD) exporting JSON/CSV.
- Store raw annotations under `data/annotations/` with metadata (annotator ID, timestamp, sample provenance).
- Maintain an evaluation changelog in `experiments/reports/`.
