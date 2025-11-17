from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..self_refine_vlm import BaseVLMAdapter


@dataclass
class HFTextInit:
    """Initialization config for a HuggingFace text-only LLM adapter.

    This adapter ignores the image input and conditions only on the question/
    answer text, which is sufficient for generating y1 and rationales when
    you primarily care about textual behavior.
    """

    model_name: str = "gpt2"
    device: Optional[str] = None
    max_new_tokens: int = 64
    temperature: float = 0.7


class HFTextAdapter(BaseVLMAdapter):
    """Adapter wrapping a HuggingFace causal LM as a baseline LLM.

    The adapter operates in text-only mode: images are ignored and prompts are
    built from question/answer strings. This is useful for quickly replacing
    the dummy adapter with a real language model for HaloQuest y1/rationale
    generation.
    """

    def __init__(self, init: HFTextInit):
        self.init = init
        self.tokenizer = AutoTokenizer.from_pretrained(init.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(init.model_name)
        device = init.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)
        self.model.eval()

    def _generate(self, prompt: str, **gen_kwargs: Any) -> str:
        tokenizer = self.tokenizer
        model = self.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        max_new_tokens = int(gen_kwargs.get("max_new_tokens", self.init.max_new_tokens))
        temperature = float(gen_kwargs.get("temperature", self.init.temperature))

        do_sample = temperature > 0.0

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )[0]

        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        # Heuristic: strip the original prompt if it is a prefix.
        if text.startswith(prompt):
            completion = text[len(prompt) :].strip()
            return completion or text.strip()
        return text.strip()

    def generate_answer(self, image: Any, question: str, **gen_kwargs: Any) -> str:
        q = (question or "").strip()
        prompt = f"Question: {q}\nAnswer:"
        return self._generate(prompt, **gen_kwargs)

    def generate_rationale(
        self, image: Any, question: str, answer: str, **gen_kwargs: Any
    ) -> str:
        q = (question or "").strip()
        a = (answer or "").strip()
        prompt = (
            f"Question: {q}\n"
            f"Answer: {a}\n"
            "Explain which parts of the image support this answer.\n"
            "Rationale:"
        )
        return self._generate(prompt, **gen_kwargs)

