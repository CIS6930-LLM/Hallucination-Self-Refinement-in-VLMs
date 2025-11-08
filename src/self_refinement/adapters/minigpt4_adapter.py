from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..self_refine_vlm import BaseVLMAdapter


def _has_minigpt4() -> bool:
    return importlib.util.find_spec("minigpt4") is not None


@dataclass
class MiniGPT4Init:
    repo_path: Optional[str] = None
    config_yaml: Optional[str] = None
    ckpt_path: Optional[str] = None
    device: Optional[str] = None
    fallback_on_missing: bool = True


class MiniGPT4Adapter(BaseVLMAdapter):
    """Adapter scaffold for MiniGPT-4.

    Notes:
      - This is a light-weight scaffold. The public MiniGPT-4 repos expose
        different APIs across versions; integrate your specific pipeline in the
        marked sections below.
      - When `fallback_on_missing=True`, the adapter returns deterministic
        placeholder outputs if MiniGPT-4 is not installed, allowing you to run
        end-to-end scaffolding without GPU/weights.
    """

    def __init__(self, init: MiniGPT4Init):
        self.init = init
        self.ready = False
        self.info = ""

        if init.repo_path:
            repo = Path(init.repo_path)
            if repo.exists():
                sys.path.insert(0, str(repo))

        if _has_minigpt4():
            # TODO: Integrate your exact MiniGPT-4 pipeline here. For example:
            #   from minigpt4.common.config import Config
            #   cfg = Config(init.config_yaml)
            #   model = load_your_minigpt4_model(cfg, ckpt=init.ckpt_path, device=init.device)
            #   self.backend = model
            #   self.ready = True
            self.info = (
                "MiniGPT-4 package detected. Adapter scaffold ready — implement model loading in adapters/minigpt4_adapter.py."
            )
        else:
            self.info = (
                "MiniGPT-4 not found. Using placeholder outputs (fallback_on_missing)."
            )
            if not init.fallback_on_missing:
                raise RuntimeError(
                    "MiniGPT-4 package not installed. Install your MiniGPT-4 repo (pip install -e .) or enable fallback_on_missing."
                )

    # --- BaseVLMAdapter interface ---
    def generate_answer(self, image: Any, question: str, **gen_kwargs) -> str:
        if self.ready:
            # Implement actual generation via MiniGPT-4 backend
            # return self.backend.generate_answer(image, question, **gen_kwargs)
            raise NotImplementedError(
                "Hook MiniGPT-4 generation call here for your repo version."
            )
        # Fallback deterministic output
        q = (question or "").strip().rstrip("?")
        return f"[MiniGPT-4 stub] Answer to: {q}"

    def generate_rationale(self, image: Any, question: str, answer: str, **gen_kwargs) -> str:
        if self.ready:
            # Implement actual rationale generation if your MiniGPT-4 supports it
            # or call the same generator with a rationale-style prompt.
            raise NotImplementedError(
                "Hook MiniGPT-4 rationale generation here or prompt the model for rationale."
            )
        return (
            f"[MiniGPT-4 stub] Rationale: The image cues support the answer '{answer}'."
        )

