from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


class BaseVLMAdapter:
    """Abstract adapter for a base VLM.

    Implementations should provide vision-language generation primitives. This
    abstraction allows swapping LLaVA, BLIP-2, MiniGPT-4, etc.
    """

    def generate_answer(self, image: Any, question: str, **gen_kwargs) -> str:
        raise NotImplementedError

    def generate_rationale(
        self, image: Any, question: str, answer: str, **gen_kwargs
    ) -> str:
        raise NotImplementedError


class DummyAdapter(BaseVLMAdapter):
    """A lightweight stand-in adapter for development and tests.

    Produces deterministic outputs without heavy dependencies.
    """

    def generate_answer(self, image: Any, question: str, **gen_kwargs) -> str:
        q = (question or "").strip().rstrip("?")
        if not q:
            return "I cannot answer without a question."
        # A simple deterministic heuristic
        return f"Hypothesis: {q.lower()} appears plausible."

    def generate_rationale(
        self, image: Any, question: str, answer: str, **gen_kwargs
    ) -> str:
        base = (question or "").strip().rstrip("?")
        return (
            "Rationale: Based on visible regions and salient objects, "
            f"the answer '{answer}' is inferred from cues related to '{base}'."
        )


@dataclass
class SelfRefineConfig:
    rationale_check_threshold: float = 0.3
    enable_revision: bool = True


class SelfRefineVLM:
    """Wraps a base VLM adapter with self-refinement capabilities.

    Stages:
      1) Initial answer generation (y1)
      2) Rationale generation (r)
      3) Self-verification and revision (y2)
    """

    def __init__(self, adapter: BaseVLMAdapter, config: Optional[SelfRefineConfig] = None):
        self.adapter = adapter
        self.config = config or SelfRefineConfig()

    def generate_answer(self, image: Any, question: str, **gen_kwargs) -> str:
        return self.adapter.generate_answer(image, question, **gen_kwargs)

    def generate_rationale(
        self, image: Any, question: str, answer: str, **gen_kwargs
    ) -> str:
        return self.adapter.generate_rationale(image, question, answer, **gen_kwargs)

    def revise_answer(
        self,
        image: Any,
        question: str,
        initial_answer: str,
        rationale: str,
        **gen_kwargs,
    ) -> str:
        """Simple self-verification and revision step.

        A placeholder implementation that checks token overlap between the
        rationale and initial answer as a proxy for consistency, and revises if
        overlap is too low.
        """

        if not self.config.enable_revision:
            return initial_answer

        a_tokens = set(t.strip(".,!?:;\"'").lower() for t in initial_answer.split())
        r_tokens = set(t.strip(".,!?:;\"'").lower() for t in rationale.split())
        a_tokens.discard("")
        r_tokens.discard("")

        if not a_tokens:
            return initial_answer

        overlap = len(a_tokens & r_tokens) / max(1, len(a_tokens))
        if overlap >= self.config.rationale_check_threshold:
            return initial_answer

        # Lightweight revision: ask adapter to regenerate conditioned on rationale
        revised_prompt = (
            f"Question: {question}\n"
            f"Initial answer: {initial_answer}\n"
            f"Rationale: {rationale}\n"
            "Given the rationale, provide a corrected, concise answer."
        )
        # For adapters that only accept (image, question), we pass revised_prompt as 'question'
        return self.adapter.generate_answer(image, revised_prompt, **gen_kwargs)


def load_adapter(name: str) -> BaseVLMAdapter:
    """Factory for VLM adapters.

    Currently returns a `DummyAdapter`. Extend with real adapters (e.g., LLaVA,
    BLIP-2) by adding imports under optional dependencies.
    """
    name = (name or "dummy").lower()
    if name in {"dummy", "mock"}:
        return DummyAdapter()

    if name in {"hf-text", "hf_text"}:
        from .adapters.hf_text_adapter import HFTextAdapter, HFTextInit

        # Use default HFTextInit; adjust model_name/device in code if needed.
        return HFTextAdapter(HFTextInit())

    # Placeholder for future extensions, e.g.:
    # if name == "llava":
    #     from .adapters.llava_adapter import LlavaAdapter
    #     return LlavaAdapter.from_pretrained(...)
    # if name == "blip2":
    #     from .adapters.blip2_adapter import Blip2Adapter
    #     return Blip2Adapter.from_pretrained(...)

    if name in {"minigpt4", "mini-gpt4", "mini_gpt4"}:
        try:
            from .adapters.minigpt4_adapter import MiniGPT4Adapter
        except Exception as e:
            raise RuntimeError(
                "MiniGPT-4 adapter import failed. Ensure the MiniGPT-4 repo is installed (pip install -e .) and dependencies are satisfied."
            ) from e
        # Return adapter without initializing heavy model; require explicit constructor usage
        raise RuntimeError(
            "Use MiniGPT4Adapter(...) directly to provide config/ckpt paths. Example: MiniGPT4Adapter(config_yaml, ckpt_path)."
        )

    raise ValueError(f"Unknown adapter: {name}")
