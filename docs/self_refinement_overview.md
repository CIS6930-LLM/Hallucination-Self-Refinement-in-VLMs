# Self-Refinement Vision-Language Model

## Overview

This repository implements a **self-refinement pipeline** for mitigating hallucinations in Vision-Language Models (VLMs) like BLIP-2 or LLaVA. The framework introduces a 3-stage reasoning loop — **Generate → Rationalize → Revise** — that forces the model to justify and verify its predictions before finalizing them.

---

## 1. Architecture Summary

### Three-Stage Pipeline

1. **Initial Generation (`y₁`):** Produce a response from the image and question.  
2. **Rationale Generation (`r`):** Explain which parts of the image support that response.  
3. **Self-Verification & Revision (`y₂`):** Re-evaluate if the rationale truly supports the claim; revise if inconsistent.

### Core Modules

- **Base VLM:** BLIP-2 / LLaVA backbone.  
- **Rationale Head:** Generates textual explanations from visual cues.  
- **Self-Check Module:** Performs consistency analysis between rationale and prediction.  
- **Contrastive Alignment Head:** Aligns image and text embeddings to reinforce factual grounding.

---

## 2. Key Code Components

```python
class SelfRefineVLM(nn.Module):
    def __init__(self, base_vlm, proj_dim=256, temperature=0.07):
        self.base = base_vlm
        self.img_proj = nn.Linear(self.base.vision_dim, proj_dim)
        self.txt_proj = nn.Linear(self.base.text_dim, proj_dim)

    def generate_answer(self, image, question):
        prompt = f"USER: {question}\nASSISTANT:"
        return self.base.generate(images=image, prompt=prompt)

    def generate_rationale(self, image, question, answer):
        prompt = f"{question}\nAnswer: {answer}\nExplain which parts of the image support this answer."
        return self.base.generate(images=image, prompt=prompt)

    def revise_answer(self, image, question, answer, rationale):
        prompt = (
            f"Question: {question}\nAnswer: {answer}\nRationale: {rationale}\n"
            "Check if the rationale supports the answer. If not, revise.\nRevised Answer:"
        )
        return self.base.generate(images=image, prompt=prompt)
```

---

## 3. Training Objectives

| Objective | Description |
| --- | --- |
| **Supervised Grounding** | Fine-tune on faithful vs hallucinated pairs with gold answers and rationales. |
| **Contrastive Consistency (InfoNCE)** | Align image and rationale embeddings while pushing apart inconsistent pairs. |
| **RLAIF** | Reinforce factual grounding using human or model-based rewards. |

### Example Loss Combination

```python
loss = (
    1.0 * loss_answer
    + 0.5 * loss_rationale
    + 0.2 * loss_contrastive
    + 0.2 * loss_hallu
)
```

---

## 4. Inference Pipeline

```python
@torch.no_grad()
def self_refine_inference(model, image, question):
    y1 = model.generate_answer(image, question)
    rationale = model.generate_rationale(image, question, y1)
    y2 = model.revise_answer(image, question, y1, rationale)
    return {"initial": y1, "rationale": rationale, "revised": y2}
```

During evaluation, compare `y₁` vs `y₂` for factual accuracy, hallucination rate, and CLIPScore.

---

## 5. Implementation Plan

| Phase | Focus | Output |
| --- | --- | --- |
| **P1** | Data curation & synthetic hallucination generation | Labeled dataset with `(y₁, r, y₂)` triplets |
| **P2** | Baseline reproduction (BLIP-2, LLaVA, SelfCheckGPT) | Baseline metrics |
| **P3** | Integrate rationale head + contrastive loss | Model v2 |
| **P4** | Add self-check + RLAIF | Model v3 |
| **P5** | Evaluate & benchmark vs ablations | Final model + report |

---

## 6. Evaluation

| Task | Metric | Baselines |
| --- | --- | --- |
| Hallucination Detection | Precision, Recall, F1, Hallucination Rate | BLIP-2, LLaVA, SelfCheckGPT |
| Rationale Faithfulness | BLEU, METEOR, Rationale-IoU | LLaVA (Grad-CAM) |
| Revision Effectiveness | Δ CLIPScore, Human Evaluation | MiniGPT-4, No-Self-Check |

---

## 7. Practical Notes

- Works with **any HuggingFace VLM** exposing `.generate()` and `.encode_*()` methods.  
- LoRA adapters can be used for lightweight fine-tuning.  
- RLAIF can be simplified to rule-based reward (factuality > 0.8 → +1).  
- Prototype can run on HiPerGator with 2×A100 (BLIP-2) or 1×B200 (LLaVA-Next).

---

## 8. Expected Outcome

- ↓ 25% hallucination rate on POPE benchmark.  
- ↑ CLIPScore and human factuality vs baseline VLMs.  
- Reusable **Self-Refine wrapper** class for multimodal reasoning.

---

### Author

**Jatin Salve**, ML Data Lead – UF CIS6930 Large Language Models (Fall 2025)
