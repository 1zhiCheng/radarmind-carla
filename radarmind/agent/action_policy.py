"""Configurable RadarMind action policy.

The policy maps high-level Agent actions such as ``slow_down`` into low-level
CARLA-style control commands. It is intentionally JSON-configurable so the mock
bridge and future real CARLA bridge can share one action contract.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[1] / "configs" / "action_policy_default.json"


@dataclass(frozen=True)
class ActionPolicyCommand:
    throttle: float
    brake: float
    steer: float
    note: str
    action_type: str
    policy_name: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ActionPolicy:
    def __init__(self, config: dict[str, Any], source_path: str | None = None):
        self.config = config
        self.source_path = source_path
        self.name = str(config.get("name", "radarmind_action_policy"))
        self.actions = config.get("actions", {})
        if not isinstance(self.actions, dict) or not self.actions:
            raise ValueError("Action policy config must define a non-empty 'actions' object")
        self.default_action = str(config.get("default_action", "keep_speed"))
        self.unknown_action = str(config.get("unknown_action", self.default_action))
        self.limits = config.get("limits", {})

    @classmethod
    def from_path(cls, path: str | Path | None = None) -> "ActionPolicy":
        policy_path = Path(path) if path else DEFAULT_POLICY_PATH
        config = json.loads(policy_path.read_text(encoding="utf-8"))
        return cls(config=config, source_path=str(policy_path))

    def _limit(self, key: str, value: Any) -> float:
        low, high = self.limits.get(key, [None, None])
        number = float(value)
        if low is not None and high is not None:
            number = clamp(number, float(low), float(high))
        return number

    def normalize_action_type(self, action: dict[str, Any] | None) -> str:
        if not action:
            return self.default_action
        raw = action.get("type") or action.get("action") or self.default_action
        action_type = str(raw)
        if action_type not in self.actions:
            return self.unknown_action if self.unknown_action in self.actions else self.default_action
        return action_type

    def command_for(self, action: dict[str, Any] | None) -> ActionPolicyCommand:
        action_type = self.normalize_action_type(action)
        spec = self.actions[action_type]
        action_reason = ""
        if isinstance(action, dict):
            action_reason = str(action.get("reason") or action.get("rationale") or "")
        note = action_reason or str(spec.get("note", action_type))
        return ActionPolicyCommand(
            throttle=self._limit("throttle", spec.get("throttle", 0.0)),
            brake=self._limit("brake", spec.get("brake", 0.0)),
            steer=self._limit("steer", spec.get("steer", 0.0)),
            note=note,
            action_type=action_type,
            policy_name=self.name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"source_path": self.source_path, **self.config}
