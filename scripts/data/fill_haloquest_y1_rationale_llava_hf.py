#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Fill y1 (baseline answer) and rationale for HaloQuest triplets "
            "using a HuggingFace LLaVA model."
        )
    )
    ap.add_argument(
        "--in_jsonl",
        type=str,
        required=True,
        help="Input JSONL with HaloQuest triplets (must include image_path and question).",
    )
    ap.add_argument(
        "--out_jsonl",
        type=str,
        required=True,
        help="Output JSONL path with y1 and rationale filled in.",
    )
    ap.add_argument(
        "--model_id",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="HuggingFace model id for LLaVA (default: llava-hf/llava-1.5-7b-hf).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string for model placement (e.g., cuda:0, cuda, cpu). "
        "Defaults to cuda if available, otherwise cpu.",
    )
    ap.add_argument(
        "--max_new_tokens_answer",
        type=int,
        default=64,
        help="Max new tokens when generating y1.",
    )
    ap.add_argument(
        "--max_new_tokens_rationale",
        type=int,
        default=96,
        help="Max new tokens when generating rationales.",
    )
    ap.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Print a simple progress message every N processed rows (0 disables).",
    )
    return ap.parse_args()


def _load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def _build_prompt(
    processor: Any,
    image: Image.Image,
    text: str,
) -> Dict[str, Any]:
    """Construct LLaVA HF inputs from an image and user text."""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful vision-language assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        },
    ]
    chat_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        images=image,
        text=chat_prompt,
        return_tensors="pt",
    )
    return inputs


def _generate(
    model: LlavaForConditionalGeneration,
    processor: Any,
    image: Image.Image,
    text: str,
    max_new_tokens: int,
) -> str:
    inputs = _build_prompt(processor, image, text)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Strip the prompt tokens and decode only the newly generated part.
    input_len = inputs["input_ids"].shape[-1]
    gen_ids = out[0, input_len:]
    text_out = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text_out


def main() -> None:
    args = parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(
        {
            "msg": "Loading LLaVA model",
            "model_id": args.model_id,
            "device": device,
        }
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        device_map=device if device.startswith("cuda") else None,
    )
    if not device.startswith("cuda"):
        model.to(device)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with in_path.open("r", encoding="utf-8") as f_in, out_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            image_path = row.get("image_path")
            question = (row.get("question") or "").strip()

            if not image_path or not question:
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += 1
                continue

            image = _load_image(image_path)

            # y1: baseline answer.
            y1_prompt = question
            y1 = _generate(
                model=model,
                processor=processor,
                image=image,
                text=y1_prompt,
                max_new_tokens=args.max_new_tokens_answer,
            )

            # Rationale: explain visual evidence for the model's own y1.
            # We intentionally do NOT use any ground-truth field (e.g., y2)
            # here, so that the rationale strictly justifies the generated y1.
            target_answer = y1
            rationale_prompt = (
                f"Question: {question}\n"
                f"Answer: {target_answer}\n"
                "Briefly explain which visual evidence in the image supports this answer."
            )
            rationale = _generate(
                model=model,
                processor=processor,
                image=image,
                text=rationale_prompt,
                max_new_tokens=args.max_new_tokens_rationale,
            )

            row["y1"] = y1
            row["rationale"] = rationale

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1

            if args.log_every and total % args.log_every == 0:
                print({"progress": total})

    print({"in": str(in_path), "out": str(out_path), "total_rows": total})


if __name__ == "__main__":
    main()
