# Self-Refinement for Hallucination Mitigation in Vision–Language Models

**Updated Research Plan — November 2025**  
**Team:** Jatin Salve (ML Data Lead), Siddharth Nahar (Modeling Lead), Sadaf Shaikh (Training Lead), Zijian Gong (Evaluation Lead)

---

## 1. Problem Motivation

Modern Vision–Language Models (VLMs) such as BLIP-2, LLaVA, and MiniGPT-4 achieve strong performance on captioning and VQA, yet hallucinate in 20–40% of responses (POPE 2023). Hallucination modes:

- **Fabricated objects:** e.g., predicting “a dog” when none exists.  
- **Incorrect relations:** e.g., describing “a woman holding a baby” when the baby is only nearby.  
- **Attribute confusions:** mislabelling color, emotion, or count.

Such errors compromise safety-critical systems (healthcare, autonomous perception, assistive AI). Current mitigation strategies focus on training-time controls (grounding alignment, contrastive losses, static RLHF) and do not dynamically detect or correct hallucinations during inference.

## 2. Core Objective

Develop a self-refinement framework that enables a VLM to:

1. Generate evidence-linked rationales justifying predictions.
2. Self-verify factual grounding between textual claims and visual evidence.
3. Iteratively revise outputs when inconsistencies or weak evidence surface.

## 3. Methodology Overview

### 3.1 Architecture

A three-stage decoding pipeline embedded in a pretrained VLM (e.g., LLaVA or BLIP-2):

1. **Initial Generation (`y₁`)** – Standard caption/VQA response.  
2. **Rationale Extraction (`r`)** – Cross-attention maps and token-level rationales showcasing the supporting evidence.  
3. **Self-Verification & Revision (`y₂`)** – A consistency module evaluates `(y₁, r)` and revises the text if confidence < τ or contradictions arise.

### 3.2 Training Objectives

| Objective | Description | Metric / Signal |
| --- | --- | --- |
| Supervised Grounding | Fine-tune on curated hallucination vs. faithful pairs | Cross-entropy + region-alignment loss |
| Contrastive Consistency | Align image–rationale embeddings; push apart inconsistent ones | InfoNCE loss |
| RLAIF | Reinforcement Learning with AI Feedback for factual grounding | Reward = factual accuracy − hallucination penalty |
| Uncertainty Regularization | Penalize high-variance tokens (MC-Dropout) | KL divergence stability loss |

### 3.3 Datasets

- **POPE Benchmark** – Object hallucination detection.  
- **COCO Captions + Synthetic Distractors** – Grounding stress test.  
- **VQA-Hallu** – QA pairs with annotated hallucinations.  
- **Synthetic Self-Refinement Corpus** – GPT-4V generated samples with `(y₁, r, y₂)` triplets.

## 4. Experimental Plan

| Experiment | Hypothesis | Metrics | Baselines |
| --- | --- | --- | --- |
| E1 – Hallucination Detection | Self-refinement reduces hallucination rate ≥ 20% vs. baseline | Precision, Recall, F1 on POPE | BLIP-2, LLaVA 1.5 |
| E2 – Rationale Faithfulness | Generated `r` correlates ≥ 0.7 with ground-truth regions | BLEU, METEOR, Rationale-IoU | LLaVA + Grad-CAM |
| E3 – Revision Effectiveness | `y₂` improves factuality without harming fluency | CLIPScore, Human Accuracy | MiniGPT-4 without loop |
| E4 – Ablation Study | Each module (rationale, contrastive, RLAIF) adds distinct gains | Δ Hallucination Rate | Ours – × component |

### Human Evaluation Protocol

- **Annotators:** Three graduate students + faculty supervisor.  
- **Samples:** 300 captions & 200 VQA pairs.  
- **Metrics:** Factuality (0–5 scale), Relevance, Confidence.  
- **Agreement Target:** Cohen’s κ ≥ 0.75.

## 5. Deliverables & Success Criteria

| Deliverable | Quantitative Target |
| --- | --- |
| Self-refinement model | ↓ Hallucination Rate ≥ 25% vs. LLaVA 1.5 on POPE |
| Rationale-grounded dataset | ≥ 50k triplets `(y₁, r, y₂)` with visual regions |
| Evaluation toolkit | Python CLI + dashboard for hallucination and rationale metrics |
| Open-source release | Training scripts, config files, and checkpoints on GitHub |
| Technical report | Benchmark results + ablation analysis for UF CIS6930 presentation |

## 6. Implementation Timeline

| Phase | Duration | Focus | Key Outputs |
| --- | --- | --- | --- |
| P1 – Data Curation & Synthetic Generation | Weeks 1–3 | Collect POPE, VQA-Hallu; generate synthetic triplets | Preprocessed dataset v1 |
| P2 – Baseline Reproduction | Weeks 4–6 | Run LLaVA 1.5 baseline on POPE | Baseline metrics + CLIPScore plots |
| P3 – Rationale Module Integration | Weeks 7–9 | Attach rationale head + supervised tuning | Intermediate checkpoint v2 |
| P4 – Self-Check & Contrastive Training | Weeks 10–12 | Introduce feedback loop + InfoNCE loss | Improved model v3 |
| P5 – RLAIF & Human Evaluation | Weeks 13–15 | Fine-tune with RLAIF; collect human feedback | Final checkpoint v4 |
| P6 – Benchmark & Release | Weeks 16–18 | Comprehensive testing, documentation | Public release + report draft |

## 7. Broader Impact & Future Work

The project promotes trustworthy multimodal AI by enabling models to explain and verify their own reasoning.

**Future extensions:**

1. Adapt self-refinement to medical imaging reports (e.g., radiology VQA).  
2. Apply the refinement loop to video VLMs for temporal consistency.  
3. Integrate verifiable reasoning with symbolic knowledge graphs (RDF/OWL).
