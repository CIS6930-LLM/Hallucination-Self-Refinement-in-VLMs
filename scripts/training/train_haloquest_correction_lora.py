#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, LlavaForConditionalGeneration, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class Example:
    image_path: str
    question: str
    y1: str
    rationale: str
    target: str


class HaloQuestCorrectionDataset(Dataset):
    """Dataset for correction SFT: (image, question, y1, rationale) -> y2."""

    def __init__(
        self,
        jsonl_path: Path,
        processor: Any,
        tokenizer: Any,
        max_samples: Optional[int] = None,
        max_length: int = 512,
        gt_field: str = "groundtruth_raw",
        use_y2_if_present: bool = True,
    ) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

        rows: List[Example] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                image_path = (row.get("image_path") or "").strip()
                question = (row.get("question") or "").strip()
                y1 = (row.get("y1") or "").strip()
                rationale = (row.get("rationale") or "").strip()

                if use_y2_if_present and row.get("y2"):
                    target = (row.get("y2") or "").strip()
                else:
                    target = (row.get(gt_field) or "").strip()

                if not image_path or not question or not target:
                    continue

                if not os.path.exists(image_path):
                    continue

                rows.append(
                    Example(
                        image_path=image_path,
                        question=question,
                        y1=y1,
                        rationale=rationale,
                        target=target,
                    )
                )
                if max_samples is not None and len(rows) >= max_samples:
                    break

        if not rows:
            raise RuntimeError(f"No valid rows found in {jsonl_path}")

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.rows[idx]

        from PIL import Image

        image = Image.open(ex.image_path).convert("RGB")

        # Build a chat-style conversation that includes an image placeholder
        # and conditions the corrected answer (target) on (question, y1, rationale).
        user_text = (
            "You are a careful vision-language assistant. "
            "Your task is to correct a possibly hallucinated answer so that it is strictly consistent with the image.\n"
            f"Question: {ex.question}\n"
            f"Initial answer (may be wrong or hallucinated): {ex.y1}\n"
            "Model rationale explaining this initial answer:\n"
            f"{ex.rationale}\n"
            "Using only information that is actually visible in the image, provide a single, short corrected answer. "
            "If the question makes assumptions that are not true in the image, explicitly say so in your answer."
        )

        # Messages for prompt-only (no assistant) and full (with assistant target).
        messages_prompt = [
            {
                "role": "system",
                "content": "You are a helpful vision-language assistant.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        messages_full = messages_prompt + [
            {
                "role": "assistant",
                "content": ex.target.strip(),
            }
        ]

        # Render chat templates to text; processor will insert image tokens.
        chat_prompt_prompt = self.processor.apply_chat_template(
            messages_prompt,
            add_generation_prompt=False,
            tokenize=False,
        )
        chat_prompt_full = self.processor.apply_chat_template(
            messages_full,
            add_generation_prompt=False,
            tokenize=False,
        )

        # Encode full sequence with image; avoid truncation here so that
        # special image tokens stay consistent with image features.
        enc = self.processor(
            images=image,
            text=chat_prompt_full,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        pixel_values = enc["pixel_values"][0]

        # Tokenize prompt-only text to estimate label masking boundary.
        prompt_ids = self.tokenizer(
            chat_prompt_prompt,
            add_special_tokens=False,
        ).input_ids
        prompt_len = len(prompt_ids)

        # Build labels: mask system+user tokens, supervise only assistant target tokens.
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="LoRA correction fine-tuning on HaloQuest (image, question, y1, rationale -> y2)."
    )
    ap.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Training JSONL with image_path, question, y1, rationale, groundtruth_raw/y2.",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save LoRA adapter and training artifacts.",
    )
    ap.add_argument(
        "--model-id",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Base LLaVA HF model id.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size.",
    )
    ap.add_argument(
        "--num-epochs",
        type=int,
        default=2,
        help="Number of training epochs.",
    )
    ap.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for AdamW.",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of training examples (for quick experiments).",
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (tokens).",
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction of data to use for validation.",
    )
    ap.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    ap.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling.",
    )
    ap.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout.",
    )
    ap.add_argument(
        "--gt-field",
        type=str,
        default="groundtruth_raw",
        help="Field to use as ground-truth answer if y2 is not present.",
    )
    ap.add_argument(
        "--no-use-y2",
        action="store_true",
        help="If set, never use y2 even when present; always use gt-field.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        {
            "msg": "Loading base LLaVA model",
            "model_id": args.model_id,
            "device": device,
        }
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )
    model.to(device)

    tokenizer = processor.tokenizer

    dataset = HaloQuestCorrectionDataset(
        jsonl_path=jsonl_path,
        processor=processor,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length,
        gt_field=args.gt_field,
        use_y2_if_present=not args.no_use_y2,
    )

    val_size = max(1, int(len(dataset) * args.val_ratio)) if len(dataset) > 1 else 0
    if val_size > 0:
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
    else:
        train_ds, val_ds = dataset, None

    print(
        {
            "num_examples_total": len(dataset),
            "num_train": len(train_ds),
            "num_val": len(val_ds) if val_ds is not None else 0,
        }
    )

    # Configure LoRA.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        if val_ds is not None
        else None
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader))
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.03 * max_train_steps),
        num_training_steps=max_train_steps,
    )

    global_step = 0
    model.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            leave=False,
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        log = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "global_step": global_step,
            "progress": round(100.0 * global_step / max_train_steps, 2),
        }

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    pixel_values = batch["pixel_values"].to(device)
                    labels = batch["labels"].to(device)

                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=labels,
                    )
                    val_loss += out.loss.item()

            avg_val_loss = val_loss / max(1, len(val_loader))
            log["val_loss"] = avg_val_loss
            model.train()

        print(log)

    # Save LoRA adapter.
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print({"msg": "Saved LoRA adapter", "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
