from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


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
    """

    def __init__(self, jsonl_path: str | Path):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._data: List[TripletExample] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                self._data.append(
                    TripletExample(
                        image_path=row.get("image_path", ""),
                        question=row.get("question", ""),
                        y1=row.get("y1", ""),
                        rationale=row.get("rationale", ""),
                        y2=row.get("y2", ""),
                    )
                )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self._data[idx]
        return {
            "image_path": ex.image_path,
            "question": ex.question,
            "y1": ex.y1,
            "rationale": ex.rationale,
            "y2": ex.y2,
        }


def collate_triplets(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    keys = ["image_path", "question", "y1", "rationale", "y2"]
    out: Dict[str, List[Any]] = {k: [] for k in keys}
    for row in batch:
        for k in keys:
            out[k].append(row.get(k))
    return out

