# Architecture Overview

## High-Level Pipeline

```text
Image / Prompt
      │
      ▼
 Pretrained VLM Encoder  ──► Initial Decoder (y₁)
      │                             │
      │                             ▼
      │                    Rationale Extractor (r)
      │                             │
      ▼                             ▼
  Evidence Encoder ─────► Consistency Module ─────► Revision Decoder (y₂)
```

1. **Initial Generation (`y₁`):** Baseline response using the pretrained decoder head.
2. **Rationale Extraction (`r`):** Cross-attention heatmaps + textual spans summarizing evidence.
3. **Consistency Module:** Evaluates `(y₁, r)` via contrastive objectives, uncertainty estimates, and checks against visual embeddings.
4. **Revision Decoder (`y₂`):** Produces revised output when confidence < τ or contradictions are detected.

## Key Components

- **Rationale Head:** Lightweight projection reading cross-attention features, supervised with region-alignment loss.  
- **Contrastive Consistency:** InfoNCE over matched vs. mismatched `(image, rationale)` pairs to tighten grounding.  
- **Uncertainty Estimator:** Monte Carlo dropout or ensemble head to assess token-level variance.  
- **Revision Policy:** Gated controller choosing between acceptance of `y₁` or re-decoding conditioned on flagged claims.

## Training Signals

- **Supervised Grounding:** Curated hallucination/faithful pairs.  
- **RLAIF:** Reward shaping with hallucination penalties.  
- **Regularization:** Stability penalty on high-variance tokens and rationale sparsity constraints.

## Implementation Notes

- Leverage modular configs under `experiments/configs/` to toggle components (rationale head, contrastive loss, RLAIF).  
- Maintain interface boundaries between `src/rationale_module`, `src/self_refinement`, and `src/evaluation` to encourage unit testing and ablation studies.  
- Export intermediate tensors for dashboard inspection (attention maps, rationale spans).
