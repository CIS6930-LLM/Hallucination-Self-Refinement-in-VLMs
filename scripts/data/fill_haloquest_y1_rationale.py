#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.self_refinement.self_refine_vlm import (  # type: ignore[import]
    SelfRefineVLM,
    SelfRefineConfig,
    load_adapter,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fill y1 (baseline answer) and rationale for HaloQuest triplets."
    )
    ap.add_argument(
        "--in_jsonl",
        type=str,
        default="/blue/cis6930/jatin.salve/Hallucination-Self-Refinement-in-VLMs/data/processed/haloquest_triplets.jsonl",
        help="Input JSONL with HaloQuest triplets (image_path, question, y2, ...).",
    )
    ap.add_argument(
        "--out_jsonl",
        type=str,
        default="/blue/cis6930/jatin.salve/Hallucination-Self-Refinement-in-VLMs/data/processed/haloquest_triplets_with_y1_rationale.jsonl",
        help="Output JSONL path with y1 and rationale filled in.",
    )
    ap.add_argument(
        "--adapter",
        type=str,
        default="dummy",
        help="Base VLM adapter name (default: dummy).",
    )
    ap.add_argument(
        "--overwrite_y1",
        action="store_true",
        help="If set, regenerate y1 even when the input row already has a non-empty y1.",
    )
    ap.add_argument(
        "--overwrite_rationale",
        action="store_true",
        help="If set, regenerate rationale even when the input row already has a non-empty rationale.",
    )
    ap.add_argument(
        "--use_y2_for_rationale",
        action="store_true",
        help=(
            "If set, use y2 (groundtruth) as the 'Answer' when prompting for rationale; "
            "otherwise, use the model's y1."
        ),
    )
    ap.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Print a simple progress message every N processed rows (0 disables).",
    )
    return ap.parse_args()


def _process_row(
    row: Dict[str, Any],
    model: SelfRefineVLM,
    overwrite_y1: bool,
    overwrite_rationale: bool,
    use_y2_for_rationale: bool,
) -> Dict[str, Any]:
    image = row.get("image_path", "")
    question = row.get("question", "") or ""

    y1_existing = (row.get("y1") or "").strip()
    if y1_existing and not overwrite_y1:
        y1 = y1_existing
    else:
        y1 = model.generate_answer(image, question)

    rationale_existing = (row.get("rationale") or "").strip()
    if rationale_existing and not overwrite_rationale:
        rationale = rationale_existing
    else:
        answer_for_r = row.get("y2", "") if use_y2_for_rationale and row.get("y2") else y1
        rationale = model.generate_rationale(image, question, answer_for_r)

    row["y1"] = y1
    row["rationale"] = rationale
    return row


def main() -> None:
    args = parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # Optional: estimate total rows for progress logging
    total_rows_estimate = 0
    if args.log_every and args.log_every > 0:
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total_rows_estimate += 1

    # Adapter: allow special handling for MiniGPT-4, otherwise use factory.
    adapter_name = (args.adapter or "dummy").lower()
    if adapter_name in {"minigpt4", "mini-gpt4", "mini_gpt4"}:
        from src.self_refinement.adapters.minigpt4_adapter import (  # type: ignore[import]
            MiniGPT4Adapter,
            MiniGPT4Init,
        )

        init = MiniGPT4Init(
            # Prefer a full upstream MiniGPT-4 checkout if present; fall back to the
            # legacy stub directory.
            repo_path=str(ROOT / "MiniGPT-4-full")
            if (ROOT / "MiniGPT-4-full").exists()
            else str(ROOT / "MiniGPT-4"),
            config_yaml=str(
                (ROOT / "MiniGPT-4-full" / "eval_configs" / "minigptv2_eval.yaml")
                if (ROOT / "MiniGPT-4-full").exists()
                else (ROOT / "MiniGPT-4" / "eval_configs" / "minigptv2_eval.yaml")
            ),
            ckpt_path=None,
            device=None,
            fallback_on_missing=True,
        )
        adapter = MiniGPT4Adapter(init)
    else:
        adapter = load_adapter(args.adapter)
    sf_model = SelfRefineVLM(adapter=adapter, config=SelfRefineConfig())

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    changed_y1 = 0
    changed_rationale = 0

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            before_y1 = row.get("y1", "")
            before_r = row.get("rationale", "")

            row = _process_row(
                row=row,
                model=sf_model,
                overwrite_y1=args.overwrite_y1,
                overwrite_rationale=args.overwrite_rationale,
                use_y2_for_rationale=args.use_y2_for_rationale,
            )

            if row.get("y1", "") != before_y1:
                changed_y1 += 1
            if row.get("rationale", "") != before_r:
                changed_rationale += 1

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1

            if args.log_every and args.log_every > 0 and total % args.log_every == 0:
                if total_rows_estimate:
                    print(
                        {
                            "progress": f"{total}/{total_rows_estimate}",
                            "percent": round(100.0 * total / total_rows_estimate, 2),
                        }
                    )
                else:
                    print({"progress": total})

    print(
        {
            "in": str(in_path),
            "out": str(out_path),
            "adapter": args.adapter,
            "total_rows": total,
            "changed_y1": changed_y1,
            "changed_rationale": changed_rationale,
        }
    )


if __name__ == "__main__":
    main()
