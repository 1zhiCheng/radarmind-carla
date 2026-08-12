"""Run Qwen2.5-VL RadarMind inference over continuous radar pseudo-image frames."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image

from radarmind.agent.action_policy import DEFAULT_POLICY_PATH
from radarmind.agent.carla_bridge import CarlaMockBridge
from radarmind.training.qwen_vl_lora_sft import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TRAIN_JSONL,
    load_processor,
    qwen_messages,
    read_jsonl,
    require_package,
)


DEFAULT_ADAPTER_PATH = "models/radarmind-carla-lora"
DEFAULT_OUTPUT = "runs/qwen_vl_agent"
DEFAULT_REGISTRY_PATH = "models/model_registry.json"


def resolve_adapter_name(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.adapter_name:
        return None
    from radarmind.evaluation.model_registry import resolve_adapter

    selected = resolve_adapter(args.registry_path, args.adapter_name)
    args.adapter_path = selected["adapter_path"]
    return selected


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelClass

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else "auto"
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if args.device_map not in {"", "none"}:
        model_kwargs["device_map"] = args.device_map
    model = ModelClass.from_pretrained(args.model_path, **model_kwargs)

    if args.adapter_path:
        require_package("peft")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)

    if args.device_map in {"", "none"}:
        model.to(args.device)
    if args.temperature <= 0 and hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    model.eval()
    return model


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Best-effort JSON extraction from a generated string."""

    stripped = text.strip()
    candidates = [stripped]
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first : last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, None
        except json.JSONDecodeError as exc:
            last_error = str(exc)
    return None, last_error if "last_error" in locals() else "no JSON object found"


def normalize_action(action: Any) -> dict[str, Any] | None:
    """Normalize imperfect model actions into the Agent action schema."""

    if isinstance(action, dict):
        action_type = action.get("type") or action.get("action") or "monitor"
        reason = action.get("reason") or action.get("rationale") or "Model produced a structured action."
        return {"type": str(action_type), "reason": str(reason)}
    if isinstance(action, str) and action.strip():
        return {"type": "monitor", "reason": action.strip()}
    return None


def render_prompt(processor: Any, record: dict[str, Any]) -> str:
    return processor.apply_chat_template(
        qwen_messages(record, include_answer=False),
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_one(
    model: torch.nn.Module,
    processor: Any,
    record: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    image = Image.open(record["radar"]["radar_image_path"]).convert("RGB")
    prompt = render_prompt(processor, record)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    if args.device_map in {"", "none"}:
        inputs = inputs.to(args.device)

    with torch.inference_mode():
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
        }
        if args.temperature > 0:
            generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
        generated_ids = model.generate(**inputs, **generation_kwargs)
    new_tokens = generated_ids[:, inputs["input_ids"].shape[1] :]
    generated_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    parsed, parse_error = extract_json_object(generated_text)

    normalized_action = normalize_action(parsed.get("recommended_action")) if parsed else None
    carla_command = None
    if args.carla == "mock" and normalized_action:
        carla_command = CarlaMockBridge(policy_path=args.action_policy).apply_action(normalized_action).to_dict()

    return {
        "sample_id": record.get("sample_id"),
        "radar": record.get("radar"),
        "prediction_text": generated_text,
        "prediction_json": parsed,
        "normalized_action": normalized_action,
        "parse_error": parse_error,
        "reference_json": record.get("radar_scene"),
        "carla_command": carla_command,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = read_jsonl(args.input_jsonl, args.max_records, object_only=args.object_only)
    processor = load_processor(args.model_path, args.min_pixels, args.max_pixels)
    model = load_model(args)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"
    started_at = time.time()
    parsed_count = 0
    with trace_path.open("w", encoding="utf-8") as fp:
        for idx, record in enumerate(records, start=1):
            row = generate_one(model, processor, record, args)
            if row["prediction_json"] is not None:
                parsed_count += 1
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {
                        "idx": idx,
                        "sample_id": row["sample_id"],
                        "parsed": row["prediction_json"] is not None,
                        "action": row.get("normalized_action"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    report = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "adapter_name": args.adapter_name,
        "adapter_registry_selection": getattr(args, "adapter_registry_selection", None),
        "action_policy": args.action_policy,
        "input_jsonl": args.input_jsonl,
        "trace_path": str(trace_path),
        "records": len(records),
        "parsed_json": parsed_count,
        "parse_rate": parsed_count / max(len(records), 1),
        "elapsed_sec": round(time.time() - started_at, 2),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--adapter-name", default="", help="Registry name, best, or best_post_training. Overrides --adapter-path when set.")
    parser.add_argument("--registry-path", default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--input-jsonl", default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-records", type=int, default=1)
    parser.add_argument("--object-only", action="store_true")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--device-map", default="")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "auto"], default="bf16")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--carla", choices=["mock", "none"], default="mock")
    parser.add_argument("--action-policy", default=str(DEFAULT_POLICY_PATH))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    selected = resolve_adapter_name(args)
    args.adapter_registry_selection = selected
    if args.device_map in {"", "none"} and args.device.startswith("cuda"):
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    run(args)


if __name__ == "__main__":
    main()
