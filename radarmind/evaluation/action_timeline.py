"""Build action timeline artifacts from a RadarMind trace.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TRACE = "/public/zzy/RadarMind/runs/qwen_vl_registry_best_post_training_smoke/trace.jsonl"
DEFAULT_OUTPUT_DIR = "/public/zzy/RadarMind/runs/action_timeline_smoke"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_action(row: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(row.get("normalized_action"), dict):
        return row["normalized_action"]
    agent_output = row.get("agent_output")
    if isinstance(agent_output, dict) and isinstance(agent_output.get("recommended_action"), dict):
        return agent_output["recommended_action"]
    prediction = row.get("prediction_json")
    if isinstance(prediction, dict) and isinstance(prediction.get("recommended_action"), dict):
        return prediction["recommended_action"]
    return None


def get_frame_id(row: dict[str, Any], idx: int) -> str:
    radar = row.get("radar") or row.get("observation") or {}
    return str(radar.get("center_frame") or radar.get("frame_id") or row.get("sample_id") or idx)


def get_command(row: dict[str, Any]) -> dict[str, Any]:
    command = row.get("carla_command")
    return command if isinstance(command, dict) else {}


def parsed(row: dict[str, Any]) -> bool:
    if "prediction_json" in row:
        return isinstance(row.get("prediction_json"), dict)
    return True


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.trace)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = output_dir / "action_timeline.jsonl"
    csv_path = output_dir / "action_timeline.csv"

    timeline: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    command_counts: Counter[str] = Counter()
    parse_count = 0
    throttle_sum = 0.0
    brake_sum = 0.0

    for idx, row in enumerate(rows, start=1):
        action = get_action(row) or {}
        command = get_command(row)
        action_type = str(action.get("type") or action.get("action") or command.get("action_type") or "missing")
        command_type = str(command.get("action_type") or action_type)
        is_parsed = parsed(row)
        parse_count += int(is_parsed)
        throttle = float(command.get("throttle", 0.0) or 0.0)
        brake = float(command.get("brake", 0.0) or 0.0)
        throttle_sum += throttle
        brake_sum += brake
        action_counts[action_type] += 1
        command_counts[command_type] += 1
        timeline.append(
            {
                "idx": idx,
                "sample_id": row.get("sample_id"),
                "frame_id": get_frame_id(row, idx),
                "parsed": is_parsed,
                "action_type": action_type,
                "action_reason": action.get("reason") or action.get("rationale") or command.get("note"),
                "command_action_type": command_type,
                "throttle": throttle,
                "brake": brake,
                "steer": float(command.get("steer", 0.0) or 0.0),
                "policy_name": command.get("policy_name"),
            }
        )

    with timeline_path.open("w", encoding="utf-8") as handle:
        for row in timeline:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "idx",
            "sample_id",
            "frame_id",
            "parsed",
            "action_type",
            "command_action_type",
            "throttle",
            "brake",
            "steer",
            "policy_name",
            "action_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in timeline:
            writer.writerow({key: row.get(key) for key in fieldnames})

    report = {
        "trace": args.trace,
        "output_dir": str(output_dir),
        "timeline_jsonl": str(timeline_path),
        "timeline_csv": str(csv_path),
        "records": len(rows),
        "parse_rate": parse_count / max(len(rows), 1),
        "action_counts": dict(action_counts),
        "command_action_counts": dict(command_counts),
        "mean_throttle": round(throttle_sum / max(len(rows), 1), 6),
        "mean_brake": round(brake_sum / max(len(rows), 1), 6),
    }
    (output_dir / "action_timeline.report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=DEFAULT_TRACE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
