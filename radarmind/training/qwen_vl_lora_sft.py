"""LoRA SFT entrypoint for RadarMind Qwen2.5-VL experiments.

The script intentionally keeps the training loop small and explicit.  It can
run a no-model dry-run to validate JSONL/image/processor formatting, and it can
also launch a real single-GPU LoRA smoke run on the local Qwen2.5-VL-3B weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_MODEL_PATH = "models/Qwen2.5-VL-3B-Instruct"
DEFAULT_TRAIN_JSONL = "data/carla_fusion/train_balanced.jsonl"
DEFAULT_VAL_JSONL = "data/carla_fusion/val.jsonl"
DEFAULT_OUTPUT_DIR = "models/radarmind-carla-lora"


def require_package(import_name: str, pip_name: str | None = None) -> None:
    try:
        __import__(import_name)
    except ImportError as exc:
        pip_name = pip_name or import_name
        raise SystemExit(
            f"Missing dependency: {import_name}. Install it with `python3 -m pip install {pip_name}`."
        ) from exc


def read_jsonl(path: str | Path, max_records: int = 0, object_only: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if object_only and not record.get("radar_scene", {}).get("objects"):
                continue
            image_path = record.get("radar", {}).get("radar_image_path")
            if not image_path or not Path(image_path).is_file():
                continue
            records.append(record)
            if max_records > 0 and len(records) >= max_records:
                break
    if not records:
        raise ValueError(f"No usable records found in {path}")
    return records


def _message_text(record: dict[str, Any], role: str) -> str:
    for message in record["messages"]:
        if message["role"] == role:
            return message["content"]
    raise KeyError(f"Record {record.get('sample_id')} has no {role!r} message")


def qwen_messages(record: dict[str, Any], include_answer: bool = True) -> list[dict[str, Any]]:
    """Convert RadarMind JSONL into Qwen2.5-VL chat messages."""

    system_text = _message_text(record, "system")
    user_text = _message_text(record, "user").replace("<image>", "").strip()
    image_path = record["radar"]["radar_image_path"]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": _message_text(record, "assistant")}],
            }
        )
    return messages


class RadarMindVLDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


@dataclass
class QwenVLCollator:
    processor: Any
    ignore_index: int = -100

    def _load_image(self, record: dict[str, Any]) -> Image.Image:
        image_path = record["radar"]["radar_image_path"]
        return Image.open(image_path).convert("RGB")

    def _render_text(self, record: dict[str, Any], include_answer: bool) -> str:
        messages = qwen_messages(record, include_answer=include_answer)
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not include_answer,
        )

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = [self._load_image(record) for record in records]
        full_texts = [self._render_text(record, include_answer=True) for record in records]
        prompt_texts = [self._render_text(record, include_answer=False) for record in records]

        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()

        for row, (prompt_text, image) in enumerate(zip(prompt_texts, images, strict=True)):
            prompt_tokens = self.processor(
                text=[prompt_text],
                images=[image],
                padding=False,
                return_tensors="pt",
            )
            prompt_len = int(prompt_tokens["input_ids"].shape[1])
            labels[row, :prompt_len] = self.ignore_index

        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = self.ignore_index

        batch["labels"] = labels
        return batch


def load_processor(model_path: str, min_pixels: int, max_pixels: int) -> Any:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True,
        local_files_only=True,
    )


def load_qwen_vl_model(args: argparse.Namespace) -> torch.nn.Module:
    from transformers import BitsAndBytesConfig

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelClass

    quantization_config = None
    if args.quantization == "4bit":
        require_package("bitsandbytes")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else "auto"
    model_kwargs = {
        "torch_dtype": dtype,
        "quantization_config": quantization_config,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if args.device_map not in {"", "none"}:
        model_kwargs["device_map"] = args.device_map
    model = ModelClass.from_pretrained(args.model_path, **model_kwargs)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def attach_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    require_package("peft")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if args.quantization == "4bit":
        model = prepare_model_for_kbit_training(model)

    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def move_batch_to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    if device == "auto":
        return batch
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def dry_run(processor: Any, records: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    collator = QwenVLCollator(processor)
    loader = DataLoader(RadarMindVLDataset(records), batch_size=batch_size, collate_fn=collator)
    batch = next(iter(loader))
    labels = batch["labels"]
    supervised_tokens = int((labels != -100).sum().item())
    return {
        "records": len(records),
        "batch_size": batch_size,
        "input_ids_shape": list(batch["input_ids"].shape),
        "attention_mask_shape": list(batch["attention_mask"].shape),
        "supervised_tokens": supervised_tokens,
        "image_grid_thw_shape": list(batch["image_grid_thw"].shape) if "image_grid_thw" in batch else None,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    train_records = read_jsonl(args.train_jsonl, args.max_train_samples, object_only=args.object_only)
    processor = load_processor(args.model_path, args.min_pixels, args.max_pixels)

    if args.dry_run:
        report = dry_run(processor, train_records, args.batch_size)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report

    require_package("peft")

    model = load_qwen_vl_model(args)
    model = attach_lora(model, args)
    model.train()

    collator = QwenVLCollator(processor)
    loader = DataLoader(
        RadarMindVLDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    device = args.device if args.device_map in {"", "none"} else "auto"
    if args.device_map in {"", "none"}:
        model.to(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    total_update_steps = args.max_steps if args.max_steps > 0 else math.ceil(len(loader) * args.epochs)
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                mean_loss = running_loss / global_step
                print(
                    json.dumps(
                        {
                            "step": global_step,
                            "total_steps": total_update_steps,
                            "epoch": epoch,
                            "loss": mean_loss,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    report = {
        "model_path": args.model_path,
        "output_dir": str(output_dir),
        "train_jsonl": args.train_jsonl,
        "train_records": len(train_records),
        "global_step": global_step,
        "mean_loss": running_loss / max(global_step, 1),
        "elapsed_sec": round(time.time() - started_at, 2),
        "lora_rank": args.lora_rank,
        "lora_target_modules": args.lora_target_modules,
    }
    (output_dir / "radarmind_lora_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-jsonl", default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--val-jsonl", default=DEFAULT_VAL_JSONL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-train-samples", type=int, default=16)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--object-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--device-map", default="", help="Use `auto` for HF device_map, empty for explicit --device.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "auto"], default="bf16")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    # Keep CUDA placement deterministic for explicit-device mode.
    if args.device_map in {"", "none"} and args.device.startswith("cuda"):
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    train(args)


if __name__ == "__main__":
    main()
