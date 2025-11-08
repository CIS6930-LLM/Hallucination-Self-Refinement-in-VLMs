from __future__ import annotations

from typing import Iterable, List, Optional, Tuple


def hallucination_rate(pred_flags: Iterable[bool]) -> float:
    """Compute fraction of predictions flagged as hallucinations.

    Args:
        pred_flags: iterable where True indicates a hallucinated output
    """
    flags = list(bool(x) for x in pred_flags)
    if not flags:
        return 0.0
    return sum(flags) / len(flags)


def iou_from_sets(a: Iterable[str], b: Iterable[str]) -> float:
    """Simple IoU over two sets of tokens/objects.

    Useful as a placeholder for rationale-object overlap when structured regions
    or boxes are not available.
    """
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / max(1, union)


def try_clip_score(
    image_paths: List[str], texts: List[str], model_name: str = "openai/clip-vit-base-patch32"
) -> Optional[List[float]]:
    """Compute CLIPScore if CLIP dependencies are available.

    Returns list of scores or None if dependencies are missing. Avoids hard
    dependency on heavy packages in the scaffold.
    """
    try:
        from PIL import Image
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except Exception:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    scores: List[float] = []
    for p, t in zip(image_paths, texts):
        image = Image.open(p).convert("RGB")
        inputs = processor(text=[t], images=[image], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
            # cosine similarities between image and text embeddings
            img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            sim = (img * txt).sum(dim=-1)
            scores.append(sim.item())
    return scores

