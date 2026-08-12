"""Build RGB + native CARLA radar-BEV SFT JSONL from privileged traces."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from radarmind.agent.live_model_policy import compose_multimodal_frame

DEFAULT_OUTPUT_DIR = "data/carla_fusion"


def _record(run_dir: Path, row: dict[str, Any], image_path: Path) -> dict[str, Any]:
    teacher = row["privileged_teacher"]
    observation = {"speed_kph": row.get("speed_kph"), "radar": row.get("radar")}
    # Put the decision first: autoregressive loss otherwise lets a long object list
    # dominate the action token that matters most to closed-loop control.
    answer = {
        "recommended_action": teacher["recommended_action"],
        "scene_summary": teacher["scene_summary"],
        "objects": [{key: obj[key] for key in ("class", "distance_m", "lateral_m",
                                                "closing_speed_mps", "ttc_s", "in_path")}
                    for obj in teacher["objects"]],
    }
    step = int(row["step"])
    return {
        "sample_id": f"{run_dir.name}_{step:06d}",
        "radar": {"radar_image_path": str(image_path), "source": "carla_rgb_native_radar_bev",
                  "representation": "composite_rgb_plus_detection_bev"},
        "messages": [
            {"role": "system", "content": "You are RadarMind, a multimodal driving agent. Fuse RGB semantics with the native CARLA radar detection BEV and return valid JSON containing objects, scene_summary, and recommended_action."},
            {"role": "user", "content": "<image>\nThe left panel is RGB and the right panel is native radar detection BEV. Telemetry: " + json.dumps(observation, ensure_ascii=False, separators=(",", ":")) + ". Infer vulnerable road users from RGB, use radar for range/closing speed, and choose keep_speed, monitor, slow_down, brake, or emergency_brake. Return JSON only."},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))},
        ],
        "radar_scene": answer,
        "metadata": {"source_run": str(run_dir), "trace_step": step, "episode": row.get("episode"),
                     "camera_path": str(run_dir / "snapshots" / f"camera_{step:06d}.jpg"),
                     "radar_bev_path": str(run_dir / "snapshots" / f"radar_bev_{step:06d}.jpg"),
                     "label_source": teacher["label_source"], "privileged_input_online": False},
    }


def collect_records(run_dirs: list[Path], images_dir: Path):
    records, skipped = [], Counter()
    images_dir.mkdir(parents=True, exist_ok=True)
    for run_dir in run_dirs:
        trace_path = run_dir / "dynamic_trace.jsonl"
        if not trace_path.is_file():
            skipped["missing_trace"] += 1
            continue
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip(): continue
                row = json.loads(line)
                if "privileged_teacher" not in row:
                    skipped["missing_teacher"] += 1; continue
                step = int(row["step"])
                camera_path = run_dir / "snapshots" / f"camera_{step:06d}.jpg"
                radar_path = run_dir / "snapshots" / f"radar_bev_{step:06d}.jpg"
                if not camera_path.is_file() or not radar_path.is_file():
                    skipped["missing_snapshot_pair"] += 1; continue
                image_path = images_dir / f"{run_dir.name}_{step:06d}.jpg"
                if not image_path.is_file():
                    compose_multimodal_frame(camera_path.read_bytes(), radar_path.read_bytes()).save(
                        image_path, format="JPEG", quality=92)
                records.append(_record(run_dir, row, image_path))
    return records, skipped


def stratified_split(records, val_ratio, seed):
    groups = defaultdict(list)
    for record in records:
        groups[record["radar_scene"]["recommended_action"]["type"]].append(record)
    rng, train, val = random.Random(seed), [], []
    for action_records in groups.values():
        rng.shuffle(action_records)
        val_count = max(1, round(len(action_records) * val_ratio)) if len(action_records) > 1 else 0
        val.extend(action_records[:val_count]); train.extend(action_records[val_count:])
    rng.shuffle(train); rng.shuffle(val)
    return train, val


def balance_records(records, seed):
    """Deterministically oversample each observed action to the majority count."""
    groups = defaultdict(list)
    for record in records:
        groups[record["radar_scene"]["recommended_action"]["type"]].append(record)
    target = max(len(group) for group in groups.values())
    balanced = []
    for group in groups.values():
        balanced.extend(group[index % len(group)] for index in range(target))
    random.Random(seed).shuffle(balanced)
    return balanced


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records: handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build(args):
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    records, skipped = collect_records([Path(path) for path in args.run_dir], output_dir / "images")
    if not records: raise ValueError("No aligned trace/snapshot records with privileged_teacher were found")
    train, val = stratified_split(records, args.val_ratio, args.seed)
    train_path, val_path = output_dir / "train.jsonl", output_dir / "val.jsonl"
    balanced = balance_records(train, args.seed)
    balanced_path = output_dir / "train_balanced.jsonl"
    _write_jsonl(train_path, train); _write_jsonl(balanced_path, balanced); _write_jsonl(val_path, val)
    actions = lambda rows: dict(Counter(r["radar_scene"]["recommended_action"]["type"] for r in rows))
    report = {"output_dir": str(output_dir), "run_dirs": args.run_dir, "records": len(records),
              "train_records": len(train), "balanced_train_records": len(balanced),
              "val_records": len(val), "train_actions": actions(train),
              "balanced_train_actions": actions(balanced), "val_actions": actions(val),
              "skipped": dict(skipped), "train_jsonl": str(train_path),
              "balanced_train_jsonl": str(balanced_path), "val_jsonl": str(val_path),
              "representation": "RGB + native CARLA RadarDetection BEV",
              "label_source": "CARLA actor ground truth + radar safety", "privileged_input_online": False}
    (output_dir / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False)); return report


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def main(): build(parse_args())
if __name__ == "__main__": main()
