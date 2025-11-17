#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, LlavaForConditionalGeneration


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    jsonl_path = (
        root
        / "data"
        / "processed"
        / "haloquest_llava_v15_full_with_rationales.jsonl"
    )
    adapter_dir = root / "outputs" / "haloquest_correction_lora_v1"
    base_id = "llava-hf/llava-1.5-7b-hf"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print({"msg": "Loading model+adapter", "device": device})

    processor = AutoProcessor.from_pretrained(adapter_dir)
    base_model = LlavaForConditionalGeneration.from_pretrained(
        base_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    base_model.to(device)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.to(device)
    model.eval()

    def run_one(example: dict) -> None:
        image_path = example.get("image_path", "")
        question = (example.get("question") or "").strip()
        y1 = (example.get("y1") or "").strip()
        rationale = (example.get("rationale") or "").strip()
        y2 = (
            (example.get("y2") or "")
            or (example.get("groundtruth_raw") or "")
        ).strip()

        if not image_path or not question:
            return

        image = Image.open(image_path).convert("RGB")

        user_text = (
            "You are a careful vision-language assistant. "
            "Your task is to correct a possibly hallucinated answer so that it is strictly consistent with the image.\n"
            f"Question: {question}\n"
            f"Initial answer (may be wrong or hallucinated): {y1}\n"
            "Model rationale explaining this initial answer:\n"
            f"{rationale}\n"
            "Using only information that is actually visible in the image, provide a single, short corrected answer. "
            "If the question makes assumptions that are not true in the image, explicitly say so in your answer."
        )

        messages = [
            {"role": "system", "content": "You are a helpful vision-language assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        chat_prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(
            images=image,
            text=chat_prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids = out_ids[0, prompt_len:]
        corrected = processor.tokenizer.decode(
            gen_ids, skip_special_tokens=True
        ).strip()

        print("Q:", question)
        print("y1 (baseline):", y1)
        print("rationale:", rationale)
        print("y2 (ground truth):", y2)
        print("corrected (LoRA):", corrected)
        print("-" * 80)

    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            run_one(ex)


if __name__ == "__main__":
    main()

