"""Self-refinement pipeline components.

Exposes the `SelfRefineVLM` wrapper and inference helper.
"""

from .self_refine_vlm import SelfRefineVLM, load_adapter
from .inference import self_refine_inference

__all__ = [
    "SelfRefineVLM",
    "load_adapter",
    "self_refine_inference",
]

