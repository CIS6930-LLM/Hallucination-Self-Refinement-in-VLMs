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


def refine_with_rationale(
    model,
    image: Any,
    question: str,
    rationale: str,
    force_revision: bool = False,
    **gen_kwargs,
) -> Dict[str, str]:
    """
    Generate an initial answer (y1), then refine it using a supplied rationale
    to obtain an improved answer (y2).

    Arguments:
        model: An object exposing generate_answer(...) and revise_answer(...)
        image: Optional vision input
        question: User question
        rationale: External rationale guiding the refinement step
        force_revision: If True, bypass the model's internal gating logic
        gen_kwargs: Passed directly to generation calls

    Returns:
        {
            "initial": <model's first answer>,
            "rationale": <the provided rationale>,
            "revised": <refined answer based on rationale>,
        }
    """

    # --- Step 1: Generate the initial response (y1) ---
    y1 = model.generate_answer(image, question, **gen_kwargs)

    # Guard against missing or useless rationale
    if not isinstance(rationale, str) or len(rationale.strip()) == 0:
        raise ValueError("Rationale must be a non-empty string.")

    # --- Step 2: Decide how refinement should be executed ---
    if force_revision:
        # Direct prompt override: no gating, no heuristics
        refinement_prompt = (
            "You are revising an answer using an explicit rationale.\n\n"
            f"Question:\n{question}\n\n"
            f"Initial Answer:\n{y1}\n\n"
            f"Given Rationale:\n{rationale}\n\n"
            "Generate a corrected final answer based strictly on the rationale. "
            "Keep the output concise and factual."
        )
        y2 = model.generate_answer(image, refinement_prompt, **gen_kwargs)

    else:
        # Delegate to the model's own refinement pipeline
        y2 = model.revise_answer(
            image=image,
            question=question,
            initial_answer=y1,
            rationale=rationale,
            **gen_kwargs,
        )

    return {
        "initial": y1,
        "rationale": rationale,
        "revised": y2,
    }

