"""Build and query the RadarMind model registry.

The registry turns per-version evaluation reports into a small model-selection
artifact. It keeps two useful winners:

- best_overall: includes the strongest SFT baseline and post-training adapters;
- best_post_training: only includes DPO/anchor/replay post-training adapters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PROJECT_MANIFEST = "/public/zzy/RadarMind/project_manifest.json"
DEFAULT_OUTPUT = "/public/zzy/RadarMind/model_registry.json"

CANDIDATES = [
    {
        "name": "carla_fusion_action_first_v0_30",
        "family": "carla_fusion_sft",
        "model_key": "qwen2_5_vl_3b_lora_carla_fusion_action_first_v0_30",
        "run_key": "carla_fusion_action_first_balanced12_v0_30",
        "doc": "PKC/docs/radarmind/VERSION_0_30_CARLA_FUSION_SFT.md",
    },
    {
        "name": "gated_joint_baseline_v0_15",
        "family": "sft",
        "model_key": "qwen2_5_vl_3b_lora_gated_joint_smoke",
        "run_key": "qwen_vl_gated_joint_agent_val50",
        "doc": "PKC/docs/radarmind/VERSION_0_15_TRAJECTORY_REWARD_MINING.md",
    },
    {
        "name": "dpo_single_pair_v0_18",
        "family": "post_training",
        "model_key": "qwen2_5_vl_3b_lora_dpo_smoke",
        "run_key": "qwen_vl_dpo_agent_val50_smoke",
        "doc": "PKC/docs/radarmind/VERSION_0_18_DPO_ADAPTER_REGRESSION.md",
    },
    {
        "name": "dpo_stabilized_4_pair_v0_19",
        "family": "post_training",
        "model_key": "qwen2_5_vl_3b_lora_dpo_stabilized_smoke",
        "run_key": "qwen_vl_dpo_stabilized_agent_val50_smoke",
        "doc": "PKC/docs/radarmind/VERSION_0_19_STABILIZED_DPO_SMOKE.md",
    },
    {
        "name": "dpo_full24_v0_20",
        "family": "post_training",
        "model_key": "qwen2_5_vl_3b_lora_dpo_full24_smoke",
        "run_key": "qwen_vl_dpo_full24_agent_val50_smoke",
        "doc": "PKC/docs/radarmind/VERSION_0_20_FULL24_DPO_STABILIZATION.md",
    },
    {
        "name": "dpo_anchor_v0_21",
        "family": "post_training",
        "model_key": "qwen2_5_vl_3b_lora_dpo_anchor_smoke",
        "run_key": "qwen_vl_dpo_anchor_agent_val50_smoke",
        "doc": "PKC/docs/radarmind/VERSION_0_21_ANCHOR_DPO_STABILIZATION.md",
    },
    {
        "name": "dpo_replay_anchor_v0_22",
        "family": "post_training",
        "model_key": "qwen2_5_vl_3b_lora_dpo_replay_anchor_smoke",
        "run_key": "qwen_vl_dpo_replay_anchor_agent_val50_smoke",
        "doc": "PKC/docs/radarmind/VERSION_0_22_REPLAY_BUFFER_MODEL_SELECTION.md",
    },
]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_candidate(run: dict[str, Any], weights: dict[str, float]) -> float:
    return round(
        weights["mean_reward"] * float(run.get("mean_reward", 0.0))
        + weights["action_accuracy"] * float(run.get("action_accuracy", 0.0))
        + weights["object_micro_f1"] * float(run.get("object_micro_f1", 0.0))
        + weights["parse_rate"] * float(run.get("parse_rate", 0.0)),
        6,
    )


def candidate_record(candidate: dict[str, str], manifest: dict[str, Any], weights: dict[str, float]) -> dict[str, Any] | None:
    models = manifest.get("models", {})
    runs = manifest.get("run_outputs", {})
    adapter_path = models.get(candidate["model_key"])
    run = runs.get(candidate["run_key"])
    if not adapter_path or not run:
        return None
    metrics = {
        "parse_rate": run.get("parse_rate", 0.0),
        "object_micro_f1": run.get("object_micro_f1", 0.0),
        "action_accuracy": run.get("action_accuracy", 0.0),
        "mean_reward": run.get("mean_reward", 0.0),
        "action_distribution": run.get("action_distribution"),
    }
    return {
        **candidate,
        "adapter_path": adapter_path,
        "run_path": run.get("path"),
        "trace": run.get("trace"),
        "eval_report": run.get("eval_report"),
        "reward_report": run.get("reward_report"),
        "metrics": metrics,
        "selection_score": score_candidate(run, weights),
    }


def rank_candidates(records: list[dict[str, Any]], min_parse_rate: float) -> list[dict[str, Any]]:
    eligible = [record for record in records if float(record["metrics"].get("parse_rate", 0.0)) >= min_parse_rate]
    return sorted(
        eligible,
        key=lambda row: (
            row["selection_score"],
            float(row["metrics"].get("mean_reward", 0.0)),
            float(row["metrics"].get("action_accuracy", 0.0)),
            float(row["metrics"].get("object_micro_f1", 0.0)),
            row["name"],
        ),
        reverse=True,
    )


def build_registry(args: argparse.Namespace) -> dict[str, Any]:
    weights = {
        "mean_reward": args.reward_weight,
        "action_accuracy": args.action_weight,
        "object_micro_f1": args.object_weight,
        "parse_rate": args.parse_weight,
    }
    manifest = load_json(args.project_manifest)
    records = [candidate_record(candidate, manifest, weights) for candidate in CANDIDATES]
    candidates = [record for record in records if record is not None]
    ranked = rank_candidates(candidates, args.min_parse_rate)
    ranked_post = rank_candidates([record for record in candidates if record["family"] == "post_training"], args.min_parse_rate)
    registry = {
        "project_manifest": args.project_manifest,
        "weights": weights,
        "min_parse_rate": args.min_parse_rate,
        "best_overall": ranked[0] if ranked else None,
        "best_post_training": ranked_post[0] if ranked_post else None,
        "candidates": ranked,
        "notes": [
            "best_overall may select the gated-joint SFT baseline if it has the highest reward-weighted score.",
            "best_post_training compares only DPO/anchor/replay adapters for post-training model selection.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "best_overall": registry["best_overall"]["name"] if registry["best_overall"] else None,
        "best_post_training": registry["best_post_training"]["name"] if registry["best_post_training"] else None,
        "candidates": len(candidates),
    }, indent=2, ensure_ascii=False))
    return registry


def resolve_adapter(registry_path: str | Path, adapter_name: str) -> dict[str, Any]:
    registry = load_json(registry_path)
    if adapter_name in {"best", "best_overall"}:
        selected = registry.get("best_overall")
    elif adapter_name == "best_post_training":
        selected = registry.get("best_post_training")
    else:
        selected = next((row for row in registry.get("candidates", []) if row.get("name") == adapter_name), None)
    if not selected:
        raise KeyError(f"No adapter named {adapter_name!r} in registry {registry_path}")
    return selected


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-manifest", default=DEFAULT_PROJECT_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--min-parse-rate", type=float, default=1.0)
    parser.add_argument("--reward-weight", type=float, default=0.5)
    parser.add_argument("--action-weight", type=float, default=0.3)
    parser.add_argument("--object-weight", type=float, default=0.2)
    parser.add_argument("--parse-weight", type=float, default=0.0)
    return parser.parse_args(argv)


def main() -> None:
    build_registry(parse_args())


if __name__ == "__main__":
    main()
