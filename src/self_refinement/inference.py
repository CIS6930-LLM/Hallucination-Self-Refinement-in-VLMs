from __future__ import annotations

from typing import Any, Dict


def self_refine_inference(model, image: Any, question: str) -> Dict[str, str]:
    """Runs the 3-stage self-refinement inference loop.

    Returns a dict with keys: "initial", "rationale", "revised".
    Caller is responsible for wrapping in `torch.no_grad()` if using torch models.
    """
    y1 = model.generate_answer(image, question)
    r = model.generate_rationale(image, question, y1)
    y2 = model.revise_answer(image, question, y1, r)
    return {"initial": y1, "rationale": r, "revised": y2}

