"""CARLA ground-truth teacher used only to label multimodal training data.

The online RadarMind policy never receives these actor states. This module
projects CARLA actors into the ego frame, derives relative motion/TTC, and
creates a compact supervision target for RGB + native radar-BEV SFT.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

VALID_ACTIONS = ("keep_speed", "monitor", "slow_down", "brake", "emergency_brake")


def _speed(vector: Any) -> float:
    return math.sqrt(float(vector.x) ** 2 + float(vector.y) ** 2 + float(vector.z) ** 2)


def actor_class(actor: Any) -> str:
    role = str(getattr(actor, "attributes", {}).get("role_name", ""))
    type_id = str(getattr(actor, "type_id", ""))
    if "pedestrian" in role or type_id.startswith("walker.pedestrian"):
        return "pedestrian"
    if "cyclist" in role or any(name in type_id for name in ("diamondback", "omafiets", "crossbike")):
        return "cyclist"
    return "vehicle"


def project_actor(ego: Any, actor: Any, *, max_range_m: float = 50.0,
                  horizontal_fov_deg: float = 90.0,
                  corridor_half_width_m: float = 2.4) -> dict[str, Any] | None:
    """Project one actor into ego coordinates and compute radial TTC."""
    if actor is None or not getattr(actor, "is_alive", True):
        return None
    ego_transform = ego.get_transform()
    actor_transform = actor.get_transform()
    dx = float(actor_transform.location.x - ego_transform.location.x)
    dy = float(actor_transform.location.y - ego_transform.location.y)
    yaw = math.radians(float(ego_transform.rotation.yaw))
    forward = math.cos(yaw) * dx + math.sin(yaw) * dy
    lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
    distance = math.hypot(dx, dy)
    azimuth_deg = math.degrees(math.atan2(lateral, forward))
    if forward <= 0.0 or distance > max_range_m or abs(azimuth_deg) > horizontal_fov_deg / 2.0:
        return None
    ego_velocity = ego.get_velocity()
    actor_velocity = actor.get_velocity()
    unit_x, unit_y = dx / max(distance, 1e-6), dy / max(distance, 1e-6)
    radial_relative_velocity = (
        (float(actor_velocity.x) - float(ego_velocity.x)) * unit_x
        + (float(actor_velocity.y) - float(ego_velocity.y)) * unit_y
    )
    closing_speed = max(0.0, -radial_relative_velocity)
    ttc = distance / closing_speed if closing_speed > 0.1 else None
    return {
        "track_id": int(actor.id), "class": actor_class(actor),
        "distance_m": round(distance, 3), "forward_m": round(forward, 3),
        "lateral_m": round(lateral, 3), "azimuth_deg": round(azimuth_deg, 3),
        "speed_mps": round(_speed(actor_velocity), 3),
        "relative_radial_velocity_mps": round(radial_relative_velocity, 3),
        "closing_speed_mps": round(closing_speed, 3),
        "ttc_s": round(ttc, 3) if ttc is not None else None,
        "in_path": abs(lateral) <= corridor_half_width_m,
    }


def _teacher_action(objects: list[dict[str, Any]], safety_action: dict[str, Any]) -> dict[str, str]:
    """Fuse privileged VRU semantics with the sensor-derived safety label."""
    severity = {name: index for index, name in enumerate(VALID_ACTIONS)}
    action_type = str(safety_action.get("type", "monitor"))
    if action_type not in severity:
        action_type = "monitor"
    reason = str(safety_action.get("reason", "Radar safety supervision."))
    path_objects = sorted((obj for obj in objects if obj["in_path"]), key=lambda obj: obj["distance_m"])
    for obj in path_objects:
        category, distance, ttc = obj["class"], float(obj["distance_m"]), obj["ttc_s"]
        proposed = "keep_speed"
        if category in {"pedestrian", "cyclist"}:
            if distance <= 8.0 and (ttc is None or ttc <= 2.5): proposed = "brake"
            elif ttc is not None and ttc <= 2.0: proposed = "emergency_brake"
            elif distance <= 16.0: proposed = "brake"
            elif distance <= 30.0 or (ttc is not None and ttc <= 4.5): proposed = "slow_down"
            else: proposed = "monitor"
        elif ttc is not None and ttc <= 1.5: proposed = "emergency_brake"
        elif distance <= 10.0: proposed = "brake"
        elif distance <= 25.0 or (ttc is not None and ttc <= 4.0): proposed = "slow_down"
        elif distance <= 40.0: proposed = "monitor"
        if severity[proposed] > severity[action_type]:
            action_type = proposed
            ttc_text = "unknown" if ttc is None else f"{ttc:.1f}s"
            reason = (f"Privileged teacher sees {category} track {obj['track_id']} "
                      f"{distance:.1f}m ahead in path (TTC={ttc_text}); choose {proposed}.")
    return {"type": action_type, "reason": reason}


def build_privileged_teacher(ego: Any, actors: Iterable[Any], safety_action: dict[str, Any], *,
                             max_range_m: float = 50.0, horizontal_fov_deg: float = 90.0,
                             corridor_half_width_m: float = 2.4) -> dict[str, Any]:
    """Create a JSON-serializable teacher target from CARLA actor ground truth."""
    objects = []
    for actor in actors:
        projected = project_actor(ego, actor, max_range_m=max_range_m,
                                  horizontal_fov_deg=horizontal_fov_deg,
                                  corridor_half_width_m=corridor_half_width_m)
        if projected is not None:
            objects.append(projected)
    objects.sort(key=lambda obj: obj["distance_m"])
    action = _teacher_action(objects, safety_action)
    path_counts = {category: sum(obj["in_path"] and obj["class"] == category for obj in objects)
                   for category in ("vehicle", "cyclist", "pedestrian")}
    nearest = objects[0] if objects else None
    scene_summary = (f"{len(objects)} labeled actors in the forward field of view; "
                     f"in-path vehicles/cyclists/pedestrians="
                     f"{path_counts['vehicle']}/{path_counts['cyclist']}/{path_counts['pedestrian']}.")
    if nearest is not None:
        scene_summary += (f" Nearest is {nearest['class']} at {nearest['distance_m']:.1f}m, "
                          f"lateral {nearest['lateral_m']:.1f}m.")
    return {"objects": objects, "scene_summary": scene_summary, "recommended_action": action,
            "label_source": "carla_actor_ground_truth_plus_radar_safety",
            "privileged_input_online": False}
