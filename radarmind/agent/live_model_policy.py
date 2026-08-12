"""Asynchronous RadarMind VLM policy for the live CARLA loop.

The worker keeps model latency and loading off the synchronous CARLA tick. It
always consumes the newest observation, so inference backlog cannot freeze the
simulator or browser stream.
"""

from __future__ import annotations

import argparse
import io
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


VALID_ACTIONS = ("keep_speed", "monitor", "slow_down", "brake", "emergency_brake")
ACTION_SEVERITY = {name: index for index, name in enumerate(VALID_ACTIONS)}
DEFAULT_MODEL_PATH = "models/Qwen2.5-VL-3B-Instruct"
DEFAULT_REGISTRY_PATH = "models/model_registry.json"


def compose_multimodal_frame(camera_jpeg: bytes, radar_jpeg: bytes) -> Image.Image:
    """Combine RGB context and the native CARLA radar BEV for VLM input."""
    camera = Image.open(io.BytesIO(camera_jpeg)).convert("RGB")
    radar = Image.open(io.BytesIO(radar_jpeg)).convert("RGB")
    target_height = 420
    camera.thumbnail((760, target_height), Image.Resampling.LANCZOS)
    radar.thumbnail((760, target_height), Image.Resampling.LANCZOS)
    header = 28
    width = camera.width + radar.width
    canvas = Image.new("RGB", (width, max(camera.height, radar.height) + header), "#06101a")
    canvas.paste(camera, (0, header))
    canvas.paste(radar, (camera.width, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 9), "RGB FOR CONTEXT", fill="#9fdcec", font=font)
    draw.text((camera.width + 10, 9), "CARLA RADAR DETECTION BEV", fill="#9fdcec", font=font)
    return canvas


def conservative_fusion(model_action: dict[str, Any] | None, safety_action: dict[str, Any]) -> dict[str, Any]:
    """Select the more conservative action; the deterministic safety shield wins ties."""
    if not model_action or model_action.get("type") not in ACTION_SEVERITY:
        return {
            **safety_action,
            "source": "safety_fallback",
            "model_action": model_action,
            "safety_action": safety_action,
        }
    model_level = ACTION_SEVERITY[model_action["type"]]
    safety_level = ACTION_SEVERITY.get(safety_action.get("type", "monitor"), 1)
    selected = model_action if model_level > safety_level else safety_action
    source = "radarmind" if model_level > safety_level else "safety_shield"
    if model_level == safety_level:
        source = "radarmind+safety_agree"
    return {
        "type": selected["type"],
        "reason": (
            f"RadarMind={model_action['type']}; safety={safety_action['type']}; "
            f"final={selected['type']}. {selected.get('reason', '')}"
        ).strip(),
        "source": source,
        "model_action": model_action,
        "safety_action": safety_action,
    }


class RadarMindPolicyWorker:
    """Latest-only asynchronous Qwen2.5-VL inference worker."""

    def __init__(
        self,
        *,
        model_path: str = DEFAULT_MODEL_PATH,
        adapter_name: str = "",
        adapter_path: str = "",
        registry_path: str = DEFAULT_REGISTRY_PATH,
        device: str = "cuda:0",
        dtype: str = "bf16",
        interval_sec: float = 2.0,
        action_ttl_sec: float = 8.0,
        max_new_tokens: int = 128,
        min_pixels: int = 128 * 28 * 28,
        max_pixels: int = 256 * 28 * 28,
    ) -> None:
        self.model_path = model_path
        self.adapter_name = adapter_name
        self.adapter_path = adapter_path
        self.registry_path = registry_path
        self.device = device
        self.dtype = dtype
        self.interval_sec = interval_sec
        self.action_ttl_sec = action_ttl_sec
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_submit = 0.0
        self._result: dict[str, Any] = {
            "status": "not_started",
            "action": None,
            "adapter_name": adapter_name,
            "adapter_path": adapter_path,
            "device": device,
        }
        self._thread = threading.Thread(target=self._run, name="radarmind-vlm-policy", daemon=True)

    def start(self) -> None:
        with self._lock:
            self._result["status"] = "loading"
            self._result["started_at"] = time.time()
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    def submit(
        self,
        *,
        step: int,
        camera_jpeg: bytes,
        radar_jpeg: bytes,
        observation: dict[str, Any],
    ) -> bool:
        now = time.monotonic()
        if now - self._last_submit < self.interval_sec:
            return False
        self._last_submit = now
        item = {
            "step": step,
            "camera_jpeg": camera_jpeg,
            "radar_jpeg": radar_jpeg,
            "observation": observation,
            "submitted_at": time.time(),
        }
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(item)
        return True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._result)
        completed_at = result.get("completed_at")
        age = time.time() - completed_at if completed_at else None
        result["age_sec"] = round(age, 3) if age is not None else None
        result["fresh"] = bool(age is not None and age <= self.action_ttl_sec and result.get("action"))
        return result

    def _set_result(self, **values: Any) -> None:
        with self._lock:
            self._result.update(values)

    def _resolve_adapter(self) -> str:
        if not self.adapter_name:
            return self.adapter_path
        from radarmind.evaluation.model_registry import resolve_adapter

        selected = resolve_adapter(self.registry_path, self.adapter_name)
        self._set_result(registry_selection=selected)
        return str(selected["adapter_path"])

    def _prompt(self, processor: Any, observation: dict[str, Any]) -> str:
        system = (
            "You are RadarMind, a radar perception and driving agent. Analyze RGB context and the "
            "CARLA radar detection BEV. Return concise valid JSON only with "
            "objects, scene_summary, and recommended_action."
        )
        telemetry = json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
        user = (
            "Analyze this live multimodal driving observation. The left image is RGB context; the right "
            "image is the native CARLA radar detection BEV, not RA/RD or raw ADC. Current telemetry is "
            f"{telemetry}. Choose recommended_action.type from keep_speed, monitor, slow_down, brake, "
            "emergency_brake. A pedestrian or cyclist in/near the path is a vulnerable road user. "
            "Use distance, closing speed and TTC; do not infer an emergency from a close but stationary "
            "return alone. Return JSON only."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "live_observation"},
                    {"type": "text", "text": user},
                ],
            },
        ]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _run(self) -> None:
        try:
            import torch
            from radarmind.inference.qwen_vl_agent import extract_json_object, load_model, normalize_action
            from radarmind.training.qwen_vl_lora_sft import load_processor

            adapter_path = self._resolve_adapter()
            model_args = argparse.Namespace(
                model_path=self.model_path,
                adapter_path=adapter_path,
                dtype=self.dtype,
                device_map="",
                device=self.device,
                temperature=0.0,
            )
            processor = load_processor(self.model_path, self.min_pixels, self.max_pixels)
            model = load_model(model_args)
            self._set_result(status="ready", adapter_path=adapter_path, loaded_at=time.time())
            while not self._stop.is_set():
                try:
                    item = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                started = time.perf_counter()
                image = compose_multimodal_frame(item["camera_jpeg"], item["radar_jpeg"])
                prompt = self._prompt(processor, item["observation"])
                inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    generated_ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
                new_tokens = generated_ids[:, inputs["input_ids"].shape[1] :]
                text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
                parsed, parse_error = extract_json_object(text)
                candidate = None
                if parsed:
                    candidate = normalize_action(parsed.get("recommended_action") or parsed.get("action"))
                if candidate and candidate.get("type") not in ACTION_SEVERITY:
                    parse_error = f"unsupported action type: {candidate.get('type')}"
                    candidate = None
                self._set_result(
                    status="ready" if candidate else "parse_error",
                    action=candidate,
                    prediction_text=text,
                    prediction_json=parsed,
                    parse_error=parse_error,
                    step=item["step"],
                    submitted_at=item["submitted_at"],
                    completed_at=time.time(),
                    latency_ms=round((time.perf_counter() - started) * 1000.0, 1),
                )
        except Exception as exc:  # keep CARLA and safety fallback alive on any model failure
            self._set_result(
                status="error",
                action=None,
                error=f"{type(exc).__name__}: {exc}",
                completed_at=time.time(),
            )
