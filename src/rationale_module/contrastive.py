from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def contrastive_infonce(
    z_img: torch.Tensor,
    z_txt: torch.Tensor,
    temperature: float = 0.07,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes symmetric InfoNCE loss between image and text embeddings.

    Args:
        z_img: (B, D) image embeddings (e.g., from a vision tower or CLIP image encoder)
        z_txt: (B, D) text/rationale embeddings
        temperature: softmax temperature
        mask: optional (B, B) mask where mask[i, j] = 0 excludes pair (i, j)

    Returns:
        loss: scalar tensor
        logits: (B, B) similarity logits after temperature scaling
    """
    assert z_img.shape == z_txt.shape, "Embeddings must have same shape"
    z_img = F.normalize(z_img, dim=-1)
    z_txt = F.normalize(z_txt, dim=-1)

    logits = (z_img @ z_txt.t()) / max(1e-8, temperature)

    if mask is not None:
        logits = logits.masked_fill(mask == 0, float("-inf"))

    labels = torch.arange(z_img.size(0), device=z_img.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    loss = 0.5 * (loss_i2t + loss_t2i)
    return loss, logits

