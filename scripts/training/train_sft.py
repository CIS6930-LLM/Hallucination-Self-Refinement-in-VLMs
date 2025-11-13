#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from src.data_pipeline.datasets import SelfRefineTripletDataset
from src.self_refinement.self_refine_vlm import (
    SelfRefineVLM,
    SelfRefineConfig,
    load_adapter,
)
from src.utils.config import load_yaml_config


@dataclass
class TrainConfig:
    """Training hyperparameters and I/O settings.

    Fields
    - batch_size: Per-step batch size for DataLoader collation.
    - epochs: Number of full passes over the training dataset.
    - lr: Base learning rate for AdamW.
    - max_seq_len: Tokenizer truncation length for input + target.
    - amp: Enable automatic mixed precision (cuda.amp) when True.
    - max_grad_norm: Gradient clipping norm value; disabled if <= 0.
    - loss_w_rationale: If > 0, include rationale supervision (CE on rationale tokens).
    - loss_w_revision: If > 0, include revision supervision when y2 targets exist.
    - out_dir: Output directory for checkpoints (created if missing).
    """
    batch_size: int = 4
    epochs: int = 1
    lr: float = 1e-5
    max_seq_len: int = 512
    amp: bool = True
    max_grad_norm: float = 1.0
    loss_w_rationale: float = 1.0
    loss_w_revision: float = 0.0
    out_dir: str = "experiments/checkpoints"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for SFT training.

    Returns
    - argparse.Namespace with fields: config, adapter, train_jsonl, val_jsonl.
    """
    ap = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) for self-refinement")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--adapter", type=str, default="dummy", help="Base VLM adapter name (e.g., dummy, llava)")
    ap.add_argument("--train_jsonl", type=str, required=True, help="Path to triplet JSONL")
    ap.add_argument("--val_jsonl", type=str, default="", help="Optional path to validation JSONL")
    return ap.parse_args()


def _resolve_train_config(cfg: Dict[str, Any]) -> TrainConfig:
    """Extract TrainConfig from a loaded YAML dictionary.

    Expects a top-level key 'train'; falls back to defaults if missing.

    Args
    - cfg: Dict loaded from YAML (see configs/self_refinement/default.yaml).

    Returns
    - TrainConfig populated from cfg["train"] or defaults.
    """
    t = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    return TrainConfig(
        batch_size=int(t.get("batch_size", 4)),
        epochs=int(t.get("epochs", 1)),
        lr=float(t.get("lr", 1e-5)),
        max_seq_len=int(t.get("max_seq_len", 512)),
        amp=bool(t.get("amp", True)),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        loss_w_rationale=float(t.get("loss_w_rationale", 1.0)),
        loss_w_revision=float(t.get("loss_w_revision", 0.0)),
        out_dir=str(t.get("out_dir", "experiments/checkpoints")),
    )


def _ensure_pad_token(tokenizer) -> None:
    """Ensure tokenizer has a pad token by mirroring eos if needed.

    Modifies tokenizer in-place to set pad_token to eos_token when pad is absent.
    """
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos = getattr(tokenizer, "eos_token", None)
        if eos is not None:
            tokenizer.pad_token = eos


def _build_rationale_prompt(question: str, y1: str) -> str:
    """Construct the rationale prompt given a question and initial answer.

    This prompt is used to supervise rationale generation with teacher forcing.
    """
    return (
        f"Question: {question}\n"
        f"Answer: {y1}\n"
        "Explain which parts of the image support this answer."
    )


def _build_revision_prompt(question: str, y1: str, rationale: str) -> str:
    """Construct the revision prompt conditioning on question, y1, and rationale.

    This is used to supervise revised answers (y2) when available.
    """
    return (
        f"Question: {question}\n"
        f"Initial answer: {y1}\n"
        f"Rationale: {rationale}\n"
        "Given the rationale, provide a corrected, concise answer."
    )


def _encode_with_labels(tokenizer, prompt: str, target: str, max_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize prompt+target and create labels masking out prompt tokens.

    Args
    - tokenizer: HF-compatible tokenizer with __call__ and pad_token_id.
    - prompt: Text that provides instruction/context; excluded from CE loss.
    - target: Gold text to supervise (rationale or revised answer); included in CE.
    - max_len: Max sequence length for truncation.

    Returns
    - input_ids: Tensor[L] token ids for prompt+target.
    - attention_mask: Tensor[L] attention mask.
    - labels: Tensor[L] where prompt tokens are -100, target tokens are ids.
    """
    # Tokenize prompt without special tokens to estimate its length in tokens
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    p_len = len(p_ids)
    enc = tokenizer(
        prompt + target,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    )
    input_ids = enc["input_ids"][0]
    attention_mask = enc["attention_mask"][0]
    labels = input_ids.clone()
    # Mask out prompt tokens; remaining tokens (target) contribute to CE loss
    mask_len = min(p_len, labels.size(0))
    labels[:mask_len] = -100
    return input_ids, attention_mask, labels


def _pad_batch(seqs: List[torch.Tensor], pad_val: int) -> torch.Tensor:
    """Left-align and pad a list of 1D tensors to a common length.

    Args
    - seqs: List of 1D tensors with possibly different lengths.
    - pad_val: Value used to pad to the max length.

    Returns
    - Tensor[B, Lmax] with each sequence placed at the start of its row.
    """
    max_len = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_val, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out


class _TripletListDataset(Dataset):
    """Thin wrapper turning a list of dict rows into a Dataset.

    Each row is expected to contain keys like: image_path, question, y1,
    rationale, y2. This wrapper allows DataLoader to iterate over the list.
    """
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        """Number of rows in the dataset."""
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return the row dict at the given index."""
        return self.rows[idx]


def _prepare_training_rows(
    ds: SelfRefineTripletDataset,
    sf_model: SelfRefineVLM,
    fill_missing_y1: bool = False,
) -> List[Dict[str, Any]]:
    """Materialize dataset examples and optionally backfill missing y1.

    Args
    - ds: SelfRefineTripletDataset instance.
    - sf_model: SelfRefineVLM used to generate y1 if backfill is enabled.
    - fill_missing_y1: When True, empty y1 fields are generated via the model.

    Returns
    - List of row dicts ready for collation/tokenization.
    """
    rows: List[Dict[str, Any]] = []
    for i in range(len(ds)):
        ex = ds[i]
        # Optionally generate y1 if missing/empty
        y1 = ex.get("y1") or ""
        if fill_missing_y1 and not y1:
            y1 = sf_model.generate_answer(ex.get("image_path"), ex.get("question", ""))
        rows.append({
            "image_path": ex.get("image_path", ""),
            "question": ex.get("question", ""),
            "y1": y1,
            "rationale": ex.get("rationale", ""),
            "y2": ex.get("y2", ""),
        })
    return rows


def _collate_sft(
    batch: List[Dict[str, Any]],
    tokenizer,
    max_len: int,
    use_revision: bool,
) -> Dict[str, torch.Tensor]:
    """Collate a batch of rows into tokenized tensors for SFT.

    Always includes rationale supervision examples; additionally appends
    revision supervision examples when use_revision is True and y2 is present.

    Returns a dict with input_ids, attention_mask, labels suitable for
    HF-style causal/seq2seq models.
    """
    input_ids_list: List[torch.Tensor] = []
    attn_mask_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    for row in batch:
        q = row.get("question", "")
        y1 = row.get("y1", "")
        r_gt = row.get("rationale", "")

        prompt = _build_rationale_prompt(q, y1)
        target = r_gt
        ids, mask, labels = _encode_with_labels(tokenizer, prompt, target, max_len)
        input_ids_list.append(ids)
        attn_mask_list.append(mask)
        labels_list.append(labels)

        if use_revision:
            # Append a second example in the same batch for revision supervision
            y2_gt = row.get("y2", "")
            if y2_gt:
                r_prompt = _build_revision_prompt(q, y1, r_gt)
                r_ids, r_mask, r_labels = _encode_with_labels(tokenizer, r_prompt, y2_gt, max_len)
                input_ids_list.append(r_ids)
                attn_mask_list.append(r_mask)
                labels_list.append(r_labels)

    pad_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else 0
    input_ids = _pad_batch(input_ids_list, pad_id)
    attention_mask = _pad_batch(attn_mask_list, 0)
    labels = _pad_batch(labels_list, -100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _has_trainable_adapter(adapter) -> bool:
    """Check whether adapter exposes trainable artifacts (model and tokenizer)."""
    return hasattr(adapter, "model") and hasattr(adapter, "tokenizer")


def train_and_validate(
    adapter,
    train_rows: List[Dict[str, Any]],
    val_rows: Optional[List[Dict[str, Any]]],
    tcfg: TrainConfig,
    device: torch.device,
) -> None:
    """Run the training loop with optional validation and checkpointing.

    Requirements
    - adapter.tokenizer: HF-compatible tokenizer.
    - adapter.model: HF-compatible model returning .loss when labels are given.

    Behavior
    - Builds DataLoaders using rationale (and optional revision) supervision.
    - Trains with AdamW + cosine schedule, AMP, and grad clipping.
    - Evaluates avg validation loss per epoch and writes checkpoints.
    """
    if not _has_trainable_adapter(adapter):
        raise RuntimeError(
            "Adapter is not trainable. Provide an adapter with 'model' and 'tokenizer' attributes compatible with HF-style training."
        )

    tokenizer = adapter.tokenizer
    _ensure_pad_token(tokenizer)
    model = adapter.model.to(device)
    model.train()

    train_ds = _TripletListDataset(train_rows)
    val_ds = _TripletListDataset(val_rows) if val_rows is not None else None

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate_sft(b, tokenizer, tcfg.max_seq_len, tcfg.loss_w_revision > 0),
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=tcfg.batch_size,
            shuffle=False,
            collate_fn=lambda b: _collate_sft(b, tokenizer, tcfg.max_seq_len, tcfg.loss_w_revision > 0),
            drop_last=False,
        )
        if val_ds is not None
        else None
    )

    optimizer = AdamW(model.parameters(), lr=tcfg.lr)
    total_steps = max(1, tcfg.epochs * len(train_loader))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=tcfg.amp)

    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=tcfg.amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            if tcfg.max_grad_norm and tcfg.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            running += float(loss.detach().cpu().item())

            if global_step % 50 == 0:
                avg = running / 50.0
                running = 0.0
                print({"epoch": epoch, "step": global_step, "train_loss": round(avg, 6)})

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss_sum += float(outputs.loss.detach().cpu().item())
                    val_count += 1
            avg_val = val_loss_sum / max(1, val_count)
            print({"epoch": epoch, "val_loss": round(avg_val, 6)})

        # Checkpoint per epoch
        ckpt_path = out_dir / f"sft_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
            },
            ckpt_path,
        )
        print({"saved": str(ckpt_path)})


def main() -> None:
    """Entry point: parse args, load data/config, and train.

    If the chosen adapter is not trainable (no model/tokenizer), performs a
    one-sample wiring check to verify generate→rationalize→revise integration.
    """
    args = parse_args()
    cfg = load_yaml_config(args.config)
    tcfg = _resolve_train_config(cfg)

    # Datasets
    train_ds = SelfRefineTripletDataset(args.train_jsonl)
    val_jsonl = args.val_jsonl or cfg.get("data", {}).get("val_jsonl", "") if isinstance(cfg, dict) else ""
    val_ds = SelfRefineTripletDataset(val_jsonl) if val_jsonl and Path(val_jsonl).exists() else None

    if len(train_ds) == 0:
        print("Dataset is empty — nothing to train.")
        return

    # Adapter and self-refine wrapper (used for optional precompute of y1)
    adapter = load_adapter(args.adapter)
    sf_model = SelfRefineVLM(adapter=adapter, config=SelfRefineConfig())

    # Prepare training/validation rows; set fill_missing_y1=True if your dataset lacks y1
    train_rows = _prepare_training_rows(train_ds, sf_model, fill_missing_y1=False)
    val_rows = _prepare_training_rows(val_ds, sf_model, fill_missing_y1=False) if val_ds is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If adapter is not trainable, fall back to a wiring check and exit
    if not _has_trainable_adapter(adapter):
        print(
            "Adapter is not trainable (no 'model'/'tokenizer'). Running a one-sample wiring check instead."
        )
        sample = train_rows[0]
        image_path = sample["image_path"]
        question = sample["question"]
        y1 = sf_model.generate_answer(image_path, question)
        r = sf_model.generate_rationale(image_path, question, y1)
        y2 = sf_model.revise_answer(image_path, question, y1, r)
        print("[SFT] Wiring check successful.")
        print({"initial": y1, "rationale": r, "revised": y2})
        return

    train_and_validate(adapter, train_rows, val_rows, tcfg, device)


if __name__ == "__main__":
    main()
