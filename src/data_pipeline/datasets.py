from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
import torch


@dataclass
class TripletExample:
    image_path: str
    question: str
    y1: str
    rationale: str
    y2: str


class SelfRefineTripletDataset:
    """JSONL dataset of (y1, rationale, y2) triplets with image and question.

    Expected JSONL keys per line:
      - image_path: str
      - question: str
      - y1: str
      - rationale: str
      - y2: str

    Parameters
    ----------
    jsonl_path:
        Path to a JSONL file containing one triplet per line.
    image_root:
        Optional root directory that is prepended to each example's image_path
        if the path is not already absolute. This is useful when JSONL files
        store relative paths but images live in a shared folder such as
        ``/blue/cis6930/jatin.salve/image/downloads``.
    load_images:
        When True, __getitem__ will load a PIL.Image for each example under the
        key ``"image"``. When False, only the image_path string is returned.
    validate_images:
        When True, the constructor checks that each resolved image path exists
        on disk and raises FileNotFoundError on the first missing image.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: Optional[str | Path] = None,
        load_images: bool = False,
        validate_images: bool = False,
    ):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.image_root = Path(image_root) if image_root is not None else None
        self.load_images = bool(load_images)

        self._data: List[TripletExample] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)

                raw_image_path = row.get("image_path", "")
                img_path = Path(raw_image_path)
                if not img_path.is_absolute() and self.image_root is not None:
                    img_path = self.image_root / img_path

                example = TripletExample(
                    image_path=str(img_path),
                    question=row.get("question", ""),
                    y1=row.get("y1", ""),
                    rationale=row.get("rationale", ""),
                    y2=row.get("y2", ""),
                )
                self._data.append(example)

        if validate_images:
            self._validate_image_paths()

    def _validate_image_paths(self) -> None:
        """Ensure all image paths exist on disk."""
        for ex in self._data:
            p = Path(ex.image_path)
            if not p.exists():
                raise FileNotFoundError(f"Missing image for dataset example: {p}")

    def __len__(self) -> int:
        return len(self._data)

    def _load_image(self, path: str) -> Image.Image:
        """Load an image as RGB."""
        img_path = Path(path)
        with Image.open(img_path) as im:
            return im.convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self._data[idx]
        out: Dict[str, Any] = {
            "image_path": ex.image_path,
            "question": ex.question,
            "y1": ex.y1,
            "rationale": ex.rationale,
            "y2": ex.y2,
        }
        if self.load_images:
            out["image"] = self._load_image(ex.image_path)
        return out


def _images_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    """Convert a list of PIL images to a (B, C, H, W) float tensor in [0, 1]."""
    if not images:
        return torch.empty(0)
    w, h = images[0].size
    for im in images:
        if im.size != (w, h):
            raise ValueError("All images must share the same size for tensor collation.")
    arr = np.stack([np.array(im, copy=False) for im in images], axis=0)  # (B, H, W, C)
    tensor = torch.from_numpy(arr).float() / 255.0
    return tensor.permute(0, 3, 1, 2).contiguous()


def collate_triplets(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of triplets.

    - Always groups scalar/string fields into lists.
    - If examples include a PIL.Image under the key ``"image"``, attempts to
      stack them into a tensor via ``_images_to_tensor``. If image sizes
      differ, returns a raw list under ``"image"`` instead.
    """
    if not batch:
        return {}

    keys = ["image_path", "question", "y1", "rationale", "y2"]
    out: Dict[str, Any] = {k: [] for k in keys}

    has_images = "image" in batch[0]
    images: List[Image.Image] = []

    for row in batch:
        for k in keys:
            out[k].append(row.get(k))
        if has_images and "image" in row and row["image"] is not None:
            images.append(row["image"])

    if has_images:
        try:
            out["image"] = _images_to_tensor(images)
        except Exception:
            out["image"] = images

    return out
