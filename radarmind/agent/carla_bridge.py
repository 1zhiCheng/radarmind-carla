"""CARLA bridge interface.

The real CARLA client can be added behind this interface. The mock bridge keeps
the replay demo useful even on machines where CARLA is not installed. v0.24
routes all mock commands through a JSON-configurable action policy so the same
policy can be reused by future real CARLA closed-loop code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from radarmind.agent.action_policy import ActionPolicy


@dataclass
class CarlaCommand:
    throttle: float
    brake: float
    steer: float
    note: str
    action_type: str = "keep_speed"
    policy_name: str = "radarmind_action_policy"

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


class CarlaMockBridge:
    def __init__(self, policy: ActionPolicy | None = None, policy_path: str | None = None):
        self.policy = policy or ActionPolicy.from_path(policy_path)

    def apply_action(self, action: dict[str, Any]) -> CarlaCommand:
        command = self.policy.command_for(action)
        return CarlaCommand(
            throttle=command.throttle,
            brake=command.brake,
            steer=command.steer,
            note=command.note,
            action_type=command.action_type,
            policy_name=command.policy_name,
        )
