"""Dynamic CARLA mixed-traffic demo using the native radar detection BEV.

This v0.30 loop deliberately keeps CARLA radar observations in their original
detection-level representation. No synthetic range-azimuth/range-Doppler array
is generated or saved. It can also record a CARLA ground-truth teacher for
offline RGB + radar-BEV supervision; privileged actor state is never submitted
to the online RadarMind policy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from radarmind.agent.action_policy import DEFAULT_POLICY_PATH, ActionPolicy, clamp
from radarmind.agent import carla_live_demo as live_base
from radarmind.agent.carla_replay import (
    choose_blueprint,
    command_to_vehicle_control,
    require_carla,
    spawn_actor,
)
from radarmind.agent.live_model_policy import (
    RadarMindPolicyWorker,
    conservative_fusion,
)
from radarmind.agent.carla_privileged_teacher import build_privileged_teacher


DEFAULT_OUTPUT_DIR = "runs/carla_closed_loop"

DYNAMIC_HTML = (
    live_base.LIVE_HTML.replace(
        "RadarMind × CARLA Live Closed Loop",
        "RadarMind × CARLA Radar BEV Loop",
    )
    .replace(
        "Radar BEV · forward 50 m",
        "CARLA native radar detection BEV · forward 50 m",
    )
    .replace(
        "Radar agent is monitoring.",
        "Radar BEV agent is monitoring dynamic traffic.",
    )
    .replace(
        "byId('points').textContent = state.radar ? state.radar.num_points : 0;",
        "byId('points').textContent = state.radar ? state.radar.num_points : 0;\n"
        "      byId('npcs').textContent = state.traffic ? "
        "state.traffic.moving_participants + '/' + state.traffic.total_participants : '0';\n"
        "      byId('policy-source').textContent = state.decision ? state.decision.source : '--';\n"
        "      byId('model-status').textContent = state.decision ? state.decision.model_status : '--';",
    )
    .replace(
        "state refresh <span id=\"updated\">--</span>",
        "moving/total traffic <span id=\"npcs\">0</span> · decision <span id=\"policy-source\">--</span> · "
        "model <span id=\"model-status\">--</span> · state refresh <span id=\"updated\">--</span>",
    )
    .replace(
        "byId('connection').textContent = 'RECONNECTING';",
        "byId('connection').textContent = 'RECONNECTING';\n"
        "      byId('action').textContent = 'TELEMETRY STALE';\n"
        "      byId('trace').textContent = 'Live server disconnected; displayed control values are stale.';",
    )
)


def clear_radar_state(live_state: live_base.LiveState, max_range: float) -> None:
    """Clear stale detections and render an empty native radar BEV."""
    empty_bev = live_base.render_radar_bev([], max_range=max_range)
    with live_state.lock:
        live_state.radar_points = []
        live_state.radar_jpeg = live_base.jpeg_bytes(empty_bev)


def cleanup_dynamic_actors(world: Any) -> int:
    actors = list(world.get_actors())
    roles = {
        "radarmind_ego",
        "radarmind_obstacle",
        "radarmind_npc",
        "radarmind_cyclist",
        "radarmind_pedestrian",
    }
    target_ids: set[int] = set()
    for actor in actors:
        try:
            if actor.attributes.get("role_name", "") in roles:
                target_ids.add(int(actor.id))
        except RuntimeError:
            continue
    for actor in actors:
        try:
            parent = actor.parent
            if parent is not None and int(parent.id) in target_ids:
                target_ids.add(int(actor.id))
        except (AttributeError, RuntimeError):
            continue
    removed = 0
    for actor in actors:
        if int(actor.id) not in target_ids:
            continue
        if actor.type_id == "controller.ai.walker":
            try:
                actor.stop()
            except RuntimeError:
                pass
        try:
            actor.destroy()
            removed += 1
        except RuntimeError:
            continue
    return removed


def spawn_lead_vehicle(
    carla: Any,
    world: Any,
    origin: Any,
    pattern: str,
    distance: float,
    rng: random.Random,
) -> Any | None:
    blueprints = list(world.get_blueprint_library().filter(pattern))
    if not blueprints:
        return None
    blueprint = rng.choice(blueprints)
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "radarmind_obstacle")
    origin_waypoint = world.get_map().get_waypoint(
        origin.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    for candidate_distance in (distance, distance + 5.0, distance - 5.0):
        for waypoint in origin_waypoint.next(max(8.0, candidate_distance)):
            transform = waypoint.transform
            transform.location.z += 0.25
            actor = world.try_spawn_actor(blueprint, transform)
            if actor is not None:
                actor.set_simulate_physics(False)
                return actor
    return None


def vehicle_blueprints(world: Any) -> list[Any]:
    result = []
    for blueprint in world.get_blueprint_library().filter("vehicle.*"):
        if blueprint.has_attribute("number_of_wheels"):
            if int(blueprint.get_attribute("number_of_wheels")) != 4:
                continue
        result.append(blueprint)
    return result


def spawn_npc_traffic(
    world: Any,
    traffic_manager: Any,
    ego_transform: Any,
    count: int,
    tm_port: int,
    rng: random.Random,
) -> list[Any]:
    blueprints = vehicle_blueprints(world)
    if not blueprints or count <= 0:
        return []

    spawn_points = list(world.get_map().get_spawn_points())
    spawn_points.sort(key=lambda transform: transform.location.distance(ego_transform.location))
    nearby = [
        transform
        for transform in spawn_points
        if 15.0 <= transform.location.distance(ego_transform.location) <= 140.0
    ]
    distant = [transform for transform in spawn_points if transform not in nearby]
    rng.shuffle(nearby)
    rng.shuffle(distant)

    actors: list[Any] = []
    for transform in nearby + distant:
        if len(actors) >= count:
            break
        blueprint = rng.choice(blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "radarmind_npc")
        if blueprint.has_attribute("color"):
            colors = blueprint.get_attribute("color").recommended_values
            if colors:
                blueprint.set_attribute("color", rng.choice(colors))
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is None:
            continue
        actor.set_autopilot(True, tm_port)
        traffic_manager.auto_lane_change(actor, True)
        traffic_manager.vehicle_percentage_speed_difference(actor, rng.uniform(-10.0, 20.0))
        actors.append(actor)
    return actors


PEDAL_BICYCLE_BLUEPRINTS = (
    "vehicle.diamondback.century",
    "vehicle.gazelle.omafiets",
    "vehicle.bh.crossbike",
)


def spawn_cyclist_traffic(
    world: Any,
    traffic_manager: Any,
    ego_transform: Any,
    count: int,
    tm_port: int,
    rng: random.Random,
) -> list[Any]:
    library = world.get_blueprint_library()
    blueprints = []
    for blueprint_id in PEDAL_BICYCLE_BLUEPRINTS:
        try:
            blueprints.append(library.find(blueprint_id))
        except RuntimeError:
            continue
    if not blueprints or count <= 0:
        return []
    spawn_points = list(world.get_map().get_spawn_points())
    spawn_points.sort(key=lambda transform: transform.location.distance(ego_transform.location))
    candidates = [
        transform for transform in spawn_points
        if 12.0 <= transform.location.distance(ego_transform.location) <= 120.0
    ]
    rng.shuffle(candidates)
    actors: list[Any] = []
    for transform in candidates:
        if len(actors) >= count:
            break
        blueprint = rng.choice(blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "radarmind_cyclist")
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is None:
            continue
        actor.set_autopilot(True, tm_port)
        traffic_manager.auto_lane_change(actor, True)
        traffic_manager.vehicle_percentage_speed_difference(actor, rng.uniform(45.0, 70.0))
        actors.append(actor)
    return actors


def spawn_pedestrian_traffic(
    carla: Any,
    world: Any,
    ego_transform: Any,
    count: int,
    running_ratio: float,
    rng: random.Random,
    synchronous: bool,
) -> tuple[list[Any], list[Any]]:
    if count <= 0:
        return [], []
    library = world.get_blueprint_library()
    walker_blueprints = list(library.filter("walker.pedestrian.*"))
    controller_blueprint = library.find("controller.ai.walker")
    walkers: list[Any] = []
    controllers: list[Any] = []
    attempts = 0
    while len(walkers) < count and attempts < count * 30:
        attempts += 1
        location = world.get_random_location_from_navigation()
        if location is None:
            continue
        distance = location.distance(ego_transform.location)
        if distance < 8.0 or distance > 110.0:
            continue
        transform = carla.Transform(location)
        blueprint = rng.choice(walker_blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "radarmind_pedestrian")
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")
        walker = world.try_spawn_actor(blueprint, transform)
        if walker is None:
            continue
        controller = world.try_spawn_actor(controller_blueprint, carla.Transform(), attach_to=walker)
        if controller is None:
            walker.destroy()
            continue
        walkers.append(walker)
        controllers.append(controller)
    if walkers:
        if synchronous:
            world.tick()
        else:
            world.wait_for_tick()
    for controller in controllers:
        try:
            controller.start()
            destination = world.get_random_location_from_navigation()
            if destination is not None:
                controller.go_to_location(destination)
            running = rng.random() < running_ratio
            controller.set_max_speed(rng.uniform(3.0, 4.5) if running else rng.uniform(1.2, 1.8))
        except RuntimeError:
            continue
    return walkers, controllers


def stop_walker_controllers(controllers: Iterable[Any]) -> None:
    for controller in controllers:
        try:
            if controller.is_alive:
                controller.stop()
        except RuntimeError:
            continue


def _actor_speeds(actors: Iterable[Any]) -> list[float]:
    speeds: list[float] = []
    for actor in actors:
        try:
            if not actor.is_alive:
                continue
            if actor.type_id.startswith("walker.pedestrian."):
                speeds.append(3.6 * abs(float(actor.get_control().speed)))
                continue
            velocity = actor.get_velocity()
        except RuntimeError:
            continue
        speeds.append(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
    return speeds


def mixed_traffic_summary(
    vehicles: Iterable[Any], cyclists: Iterable[Any], pedestrians: Iterable[Any]
) -> dict[str, Any]:
    vehicle_speeds = _actor_speeds(vehicles)
    cyclist_speeds = _actor_speeds(cyclists)
    pedestrian_speeds = _actor_speeds(pedestrians)
    speeds = vehicle_speeds + cyclist_speeds + pedestrian_speeds
    return {
        "total_vehicles": len(vehicle_speeds),
        "moving_vehicles": sum(speed > 1.0 for speed in vehicle_speeds),
        "total_cyclists": len(cyclist_speeds),
        "moving_cyclists": sum(speed > 1.0 for speed in cyclist_speeds),
        "total_pedestrians": len(pedestrian_speeds),
        "moving_pedestrians": sum(speed > 0.3 for speed in pedestrian_speeds),
        "total_participants": len(speeds),
        "moving_participants": (
            sum(speed > 1.0 for speed in vehicle_speeds + cyclist_speeds)
            + sum(speed > 0.3 for speed in pedestrian_speeds)
        ),
        "mean_speed_kph": round(sum(speeds) / len(speeds), 3) if speeds else 0.0,
        "max_speed_kph": round(max(speeds), 3) if speeds else 0.0,
    }


def route_steer(carla: Any, world: Any, vehicle: Any, lookahead: float, gain: float, max_steer: float) -> float:
    transform = vehicle.get_transform()
    waypoint = world.get_map().get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    candidates = waypoint.next(lookahead)
    if not candidates:
        return 0.0
    target = candidates[0].transform.location
    dx = target.x - transform.location.x
    dy = target.y - transform.location.y
    heading = math.radians(transform.rotation.yaw)
    local_x = math.cos(heading) * dx + math.sin(heading) * dy
    local_y = -math.sin(heading) * dx + math.cos(heading) * dy
    heading_error = math.atan2(local_y, max(0.1, local_x))
    return clamp(gain * heading_error, -max_steer, max_steer)


def radar_corridor_summary(
    points: list[dict[str, float]],
    corridor_half_width: float,
    radar_height: float,
    radar_pitch_deg: float,
    ground_clearance: float,
) -> dict[str, Any]:
    corridor_points = []
    ground_filtered_points = 0
    for point in points:
        world_elevation = point["altitude"] + math.radians(radar_pitch_deg)
        horizontal_depth = point["depth"] * math.cos(world_elevation)
        lateral = horizontal_depth * math.sin(point["azimuth"])
        forward = horizontal_depth * math.cos(point["azimuth"])
        estimated_height = radar_height + point["depth"] * math.sin(world_elevation)
        if estimated_height < ground_clearance:
            ground_filtered_points += 1
            continue
        if forward > 0.0 and abs(lateral) <= corridor_half_width:
            corridor_points.append(point)
    summary = live_base.radar_summary(corridor_points)
    summary["num_points"] = len(points)
    summary["corridor_points"] = len(corridor_points)
    summary["ground_filtered_points"] = ground_filtered_points
    if corridor_points:
        nearest = min(corridor_points, key=lambda point: point["depth"])
        summary["nearest_detection"] = {
            "depth_m": round(nearest["depth"], 3),
            "azimuth_deg": round(math.degrees(nearest["azimuth"]), 3),
            "altitude_deg": round(math.degrees(nearest["altitude"]), 3),
            "lateral_m": round(nearest["depth"] * math.sin(nearest["azimuth"]), 3),
            "estimated_height_m": round(
                radar_height
                + nearest["depth"]
                * math.sin(nearest["altitude"] + math.radians(radar_pitch_deg)),
                3,
            ),
            "relative_velocity_mps": round(nearest["velocity"], 3),
        }
    else:
        summary["nearest_detection"] = None
    return summary


def driving_action(
    summary: dict[str, Any],
    args: argparse.Namespace,
    ego_speed_kph: float,
) -> dict[str, str]:
    depth = summary.get("min_depth_m")
    if depth is None:
        summary["closing_speed_mps"] = 0.0
        summary["ttc_sec"] = None
        return {
            "type": "keep_speed",
            "reason": "No above-ground obstacle in the ego-lane radar corridor; continue route following.",
        }

    relative_velocity = float(summary.get("nearest_velocity_mps") or 0.0)
    closing_speed = max(0.0, -relative_velocity)
    ttc = float(depth) / closing_speed if closing_speed >= 0.05 else None
    summary["closing_speed_mps"] = round(closing_speed, 3)
    summary["ttc_sec"] = round(ttc, 3) if ttc is not None else None

    if float(depth) <= args.emergency_range:
        is_emergency = (
            closing_speed >= args.emergency_min_closing_speed_mps
            and ttc is not None
            and ttc <= args.emergency_ttc_seconds
        )
        if is_emergency:
            return {
                "type": "emergency_brake",
                "reason": (
                    f"Collision risk: range={depth:.1f} m, closing={closing_speed:.1f} m/s, "
                    f"TTC={ttc:.1f} s."
                ),
            }
        return {
            "type": "brake",
            "reason": (
                f"Close target at {depth:.1f} m, but no emergency closing condition "
                f"(closing={closing_speed:.1f} m/s, ego={ego_speed_kph:.1f} km/h); controlled hold."
            ),
        }
    if float(depth) <= args.brake_range:
        return {
            "type": "brake",
            "reason": f"Nearest radar return is {depth:.1f} m: controlled braking/headway hold.",
        }
    if float(depth) <= args.slow_range:
        return {
            "type": "slow_down",
            "reason": f"Nearest radar return is {depth:.1f} m: reduce speed and monitor TTC.",
        }
    return {
        "type": "keep_speed",
        "reason": f"Nearest radar return is {depth:.1f} m: forward corridor is currently safe.",
    }


def action_target_speed_kph(action_type: str, args: argparse.Namespace) -> float:
    return {
        "keep_speed": args.cruise_speed_kph,
        "monitor": args.monitor_speed_kph,
        "slow_down": args.slow_speed_kph,
        "brake": args.brake_speed_kph,
        "emergency_brake": 0.0,
    }.get(action_type, args.monitor_speed_kph)


def select_closed_loop_action(
    policy_mode: str,
    model_state: dict[str, Any],
    safety_action: dict[str, Any],
) -> dict[str, Any]:
    if policy_mode == "safety":
        return {
            **safety_action,
            "source": "safety_only",
            "model_action": None,
            "safety_action": safety_action,
        }
    model_action = model_state.get("action") if model_state.get("fresh") else None
    if policy_mode == "hybrid":
        return conservative_fusion(model_action, safety_action)
    if not model_action:
        return {
            **safety_action,
            "source": "safety_fallback_model_unavailable",
            "model_action": None,
            "safety_action": safety_action,
        }
    if safety_action.get("type") == "emergency_brake" and model_action.get("type") != "emergency_brake":
        return {
            **safety_action,
            "source": "emergency_safety_override",
            "model_action": model_action,
            "safety_action": safety_action,
        }
    return {
        **model_action,
        "source": "radarmind",
        "model_action": model_action,
        "safety_action": safety_action,
    }


def zero_vehicle_motion(carla: Any, vehicle: Any) -> None:
    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))


def stop_sensor_safely(sensor: Any | None) -> None:
    if sensor is None:
        return
    try:
        if sensor.is_alive:
            sensor.stop()
    except RuntimeError:
        return


def destroy_actors_safely(client: Any, carla: Any, actors: Iterable[Any | None]) -> int:
    actor_ids: list[int] = []
    for actor in actors:
        if actor is None:
            continue
        try:
            if actor.is_alive:
                actor_ids.append(int(actor.id))
        except RuntimeError:
            continue
    if actor_ids:
        client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in actor_ids])
    return len(actor_ids)


def finite_control_value(value: Any) -> float:
    number = float(value)
    return number if math.isfinite(number) else 0.0


def run(args: argparse.Namespace) -> dict[str, Any]:
    carla = require_carla()
    live_base.LIVE_HTML = DYNAMIC_HTML
    output_dir = Path(args.output_dir)
    snapshots_dir = output_dir / "snapshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "dynamic_trace.jsonl"
    report_path = output_dir / "dynamic_demo.report.json"

    live_state = live_base.LiveState(
        camera_jpeg=live_base.placeholder_image(
            (args.image_width, args.image_height), "Waiting for RGB camera"
        ),
        radar_jpeg=live_base.placeholder_image((600, 600), "Waiting for CARLA radar detections"),
    )

    rng = random.Random(args.seed)
    policy = ActionPolicy.from_path(args.action_policy)
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_random_device_seed(args.seed)
    dashboard = live_base.start_dashboard(args.web_host, args.web_port, live_state)
    model_worker: RadarMindPolicyWorker | None = None
    if args.policy_mode != "safety":
        model_worker = RadarMindPolicyWorker(
            model_path=args.model_path,
            adapter_name=args.model_adapter_name,
            adapter_path=args.model_adapter_path,
            registry_path=args.model_registry,
            device=args.model_device,
            dtype=args.model_dtype,
            interval_sec=args.model_interval_sec,
            action_ttl_sec=args.model_action_ttl_sec,
            max_new_tokens=args.model_max_new_tokens,
        )
        model_worker.start()

    vehicle = None
    camera = None
    radar = None
    obstacle = None
    npc_actors: list[Any] = []
    cyclist_actors: list[Any] = []
    pedestrian_actors: list[Any] = []
    pedestrian_controllers: list[Any] = []
    cleaned_actors = 0
    records = 0
    snapshots_saved = 0
    max_moving_participants = 0
    max_traffic_speed_kph = 0.0
    max_ego_speed_kph = 0.0
    total_distance_m = 0.0
    episode_distance_m = 0.0
    previous_location = None
    episode = 1
    episode_step = 0
    stopped_steps = 0
    started_at = time.time()
    interrupted = False

    try:
        if args.sync:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = args.fixed_delta_seconds
            world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

        if args.cleanup_existing:
            cleaned_actors = cleanup_dynamic_actors(world)
            if args.sync:
                world.tick()

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("CARLA map has no spawn points")
        spawn_transform = spawn_points[args.spawn_index % len(spawn_points)]
        vehicle_bp = choose_blueprint(world, args.vehicle_filter, rng)
        vehicle = spawn_actor(world, vehicle_bp, spawn_transform, retries=len(spawn_points))
        previous_location = vehicle.get_location()

        if args.spawn_obstacle:
            obstacle = spawn_lead_vehicle(
                carla,
                world,
                vehicle.get_transform(),
                pattern=args.obstacle_filter,
                distance=args.obstacle_distance,
                rng=rng,
            )

        npc_actors = spawn_npc_traffic(
            world,
            traffic_manager,
            ego_transform=vehicle.get_transform(),
            count=args.npc_count,
            tm_port=args.tm_port,
            rng=rng,
        )
        cyclist_actors = spawn_cyclist_traffic(
            world,
            traffic_manager,
            ego_transform=vehicle.get_transform(),
            count=args.cyclist_count,
            tm_port=args.tm_port,
            rng=rng,
        )
        world.set_pedestrians_cross_factor(args.pedestrian_crossing_factor)
        pedestrian_actors, pedestrian_controllers = spawn_pedestrian_traffic(
            carla,
            world,
            ego_transform=vehicle.get_transform(),
            count=args.pedestrian_count,
            running_ratio=args.pedestrian_running_ratio,
            rng=rng,
            synchronous=args.sync,
        )

        if args.ego_controller == "traffic_manager":
            vehicle.set_autopilot(True, args.tm_port)
            traffic_manager.auto_lane_change(vehicle, True)
            traffic_manager.distance_to_leading_vehicle(vehicle, args.follow_distance_m)
            traffic_manager.ignore_lights_percentage(vehicle, 0.0)
            traffic_manager.ignore_signs_percentage(vehicle, 0.0)
            traffic_manager.set_desired_speed(vehicle, args.cruise_speed_kph)

        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(args.image_width))
        camera_bp.set_attribute("image_size_y", str(args.image_height))
        camera_bp.set_attribute("fov", str(args.camera_fov))
        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(
                carla.Location(x=args.camera_x, z=args.camera_z),
                carla.Rotation(pitch=args.camera_pitch),
            ),
            attach_to=vehicle,
        )
        camera.listen(live_state.update_camera)

        radar_bp = world.get_blueprint_library().find("sensor.other.radar")
        radar_bp.set_attribute("horizontal_fov", str(args.radar_horizontal_fov))
        radar_bp.set_attribute("vertical_fov", str(args.radar_vertical_fov))
        radar_bp.set_attribute("range", str(args.radar_range))
        radar_bp.set_attribute("points_per_second", str(args.radar_points_per_second))
        radar_bp.set_attribute("sensor_tick", str(args.fixed_delta_seconds))
        radar = world.spawn_actor(
            radar_bp,
            carla.Transform(
                carla.Location(x=args.radar_x, z=args.radar_z),
                carla.Rotation(pitch=args.radar_pitch),
            ),
            attach_to=vehicle,
        )
        radar.listen(
            lambda measurement: live_state.update_radar(
                measurement,
                max_range=args.radar_range,
            )
        )

        print(
            json.dumps(
                {
                    "dashboard": f"http://{args.web_host}:{args.web_port}",
                    "local_tunnel": f"ssh -N -L 17860:127.0.0.1:{args.web_port} {args.ssh_user}@{args.ssh_host}",
                    "mixed_traffic": {
                        "vehicles": len(npc_actors),
                        "cyclists": len(cyclist_actors),
                        "pedestrians": len(pedestrian_actors),
                    },
                    "policy_mode": args.policy_mode,
                    "model_device": args.model_device if model_worker else None,
                    "model_adapter_name": args.model_adapter_name if model_worker else None,
                    "radar_representation": "native CARLA RadarDetection BEV",
                    "output_dir": str(output_dir),
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )

        with trace_path.open("w", encoding="utf-8") as trace_handle:
            step_idx = 0
            while args.max_steps == 0 or step_idx < args.max_steps:
                wall_step_start = time.perf_counter()
                step_idx += 1
                episode_step += 1
                frame_id = world.tick() if args.sync else None
                if not args.sync:
                    time.sleep(args.step_sec)

                current_speed = live_base.speed_kph(vehicle)
                camera_jpeg, radar_jpeg, _, points = live_state.snapshot()
                summary = radar_corridor_summary(
                    points,
                    corridor_half_width=args.corridor_half_width,
                    radar_height=args.radar_z,
                    radar_pitch_deg=args.radar_pitch,
                    ground_clearance=args.radar_ground_clearance,
                )
                safety_action = driving_action(summary, args, current_speed)
                traffic = mixed_traffic_summary(npc_actors, cyclist_actors, pedestrian_actors)
                privileged_teacher = build_privileged_teacher(
                    vehicle,
                    [*npc_actors, *cyclist_actors, *pedestrian_actors, obstacle],
                    safety_action,
                    max_range_m=max(args.radar_range, 50.0),
                    horizontal_fov_deg=args.camera_fov,
                    corridor_half_width_m=args.corridor_half_width,
                )
                if model_worker is not None:
                    model_worker.submit(
                        step=step_idx,
                        camera_jpeg=camera_jpeg,
                        radar_jpeg=radar_jpeg,
                        observation={
                            "speed_kph": round(current_speed, 2),
                            "radar": summary,
                        },
                    )
                    model_state = model_worker.snapshot()
                else:
                    model_state = {"status": "disabled", "action": None, "fresh": False}
                action = select_closed_loop_action(args.policy_mode, model_state, safety_action)
                policy_command = policy.command_for(action).to_dict()
                target_speed_kph = action_target_speed_kph(action["type"], args)
                if args.ego_controller == "traffic_manager":
                    traffic_manager.set_desired_speed(vehicle, target_speed_kph)
                    control = vehicle.get_control()
                else:
                    steer = route_steer(
                        carla,
                        world,
                        vehicle,
                        lookahead=args.route_lookahead,
                        gain=args.steer_gain,
                        max_steer=args.max_steer,
                    )
                    policy_command["steer"] = steer
                    control = command_to_vehicle_control(carla, policy_command)
                    vehicle.apply_control(control)

                max_ego_speed_kph = max(max_ego_speed_kph, current_speed)
                vehicle_transform = vehicle.get_transform()
                current_location = vehicle_transform.location
                step_distance = current_location.distance(previous_location) if previous_location is not None else 0.0
                if step_distance < 5.0:
                    total_distance_m += step_distance
                    episode_distance_m += step_distance
                previous_location = current_location
                max_moving_participants = max(
                    max_moving_participants, int(traffic["moving_participants"])
                )
                max_traffic_speed_kph = max(max_traffic_speed_kph, float(traffic["max_speed_kph"]))
                if current_speed < args.stopped_speed_kph and action["type"] in {"brake", "emergency_brake"}:
                    stopped_steps += 1
                else:
                    stopped_steps = 0

                state = {
                    "status": "live",
                    "step": step_idx,
                    "episode": episode,
                    "episode_step": episode_step,
                    "carla_frame": int(frame_id) if frame_id is not None else None,
                    "timestamp": time.time(),
                    "map": world.get_map().name,
                    "action": action["type"],
                    "reason": action["reason"],
                    "decision": {
                        "policy_mode": args.policy_mode,
                        "source": action.get("source"),
                        "model_action": action.get("model_action"),
                        "safety_action": safety_action,
                        "final_action": {"type": action["type"], "reason": action["reason"]},
                        "model_status": model_state.get("status"),
                        "model_fresh": model_state.get("fresh", False),
                        "model_step": model_state.get("step"),
                        "model_age_sec": model_state.get("age_sec"),
                        "model_latency_ms": model_state.get("latency_ms"),
                        "model_adapter": model_state.get("adapter_path"),
                        "model_prediction_json": model_state.get("prediction_json"),
                        "model_parse_error": model_state.get("parse_error"),
                        "model_error": model_state.get("error"),
                    },
                    "radar": summary,
                    "radar_representation": {
                        "source": "native CARLA RadarDetection polar points rendered as BEV",
                        "fields": ["depth", "azimuth", "altitude", "velocity"],
                        "not_generated": ["range_azimuth", "range_doppler", "raw_adc"],
                    },
                    "speed_kph": round(current_speed, 3),
                    "target_speed_kph": round(target_speed_kph, 3),
                    "ego_controller": args.ego_controller,
                    "action_policy_command": policy_command,
                    "vehicle": {
                        "x": round(vehicle_transform.location.x, 3),
                        "y": round(vehicle_transform.location.y, 3),
                        "z": round(vehicle_transform.location.z, 3),
                        "yaw": round(vehicle_transform.rotation.yaw, 3),
                        "episode_distance_m": round(episode_distance_m, 3),
                        "total_distance_m": round(total_distance_m, 3),
                    },
                    "control": {
                        "throttle": round(finite_control_value(control.throttle), 3),
                        "brake": round(finite_control_value(control.brake), 3),
                        "steer": round(finite_control_value(control.steer), 3),
                    },
                    "traffic": traffic,
                    "privileged_teacher": privileged_teacher,
                    "agent": "radarmind_fusion_teacher_agent_v0_30",
                }
                with live_state.lock:
                    live_state.state = state
                trace_handle.write(json.dumps(state, ensure_ascii=False) + "\n")
                trace_handle.flush()
                records += 1

                if args.save_every > 0 and step_idx % args.save_every == 0:
                    snapshots_dir.mkdir(parents=True, exist_ok=True)
                    (snapshots_dir / f"camera_{step_idx:06d}.jpg").write_bytes(camera_jpeg)
                    (snapshots_dir / f"radar_bev_{step_idx:06d}.jpg").write_bytes(radar_jpeg)
                    snapshots_saved += 2

                if args.auto_reset and stopped_steps >= args.reset_stopped_steps:
                    zero_vehicle_motion(carla, vehicle)
                    vehicle.set_transform(spawn_transform)
                    zero_vehicle_motion(carla, vehicle)
                    clear_radar_state(live_state, args.radar_range)
                    previous_location = spawn_transform.location
                    episode_distance_m = 0.0
                    episode += 1
                    episode_step = 0
                    stopped_steps = 0

                if args.realtime:
                    elapsed = time.perf_counter() - wall_step_start
                    time.sleep(max(0.0, args.fixed_delta_seconds - elapsed))
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if model_worker is not None:
            model_worker.stop()
        dashboard.shutdown()
        dashboard.server_close()
        stop_sensor_safely(radar)
        stop_sensor_safely(camera)
        stop_walker_controllers(pedestrian_controllers)
        destroy_actors_safely(
            client,
            carla,
            [
                radar,
                camera,
                *pedestrian_controllers,
                *pedestrian_actors,
                *cyclist_actors,
                *npc_actors,
                obstacle,
                vehicle,
            ],
        )
        if args.sync:
            traffic_manager.set_synchronous_mode(False)
            world.apply_settings(original_settings)

    report = {
        "output_dir": str(output_dir),
        "trace_path": str(trace_path),
        "report_path": str(report_path),
        "dashboard_url": f"http://{args.web_host}:{args.web_port}",
        "map": world.get_map().name,
        "records": records,
        "episodes": episode,
        "npc_vehicles": len(npc_actors),
        "cyclists": len(cyclist_actors),
        "pedestrians": len(pedestrian_actors),
        "max_moving_participants": max_moving_participants,
        "max_traffic_speed_kph": round(max_traffic_speed_kph, 3),
        "max_ego_speed_kph": round(max_ego_speed_kph, 3),
        "total_ego_distance_m": round(total_distance_m, 3),
        "obstacle_spawned": obstacle is not None,
        "snapshots_saved": snapshots_saved,
        "radar_representation": "native CARLA RadarDetection polar points rendered as BEV",
        "generated_ra_rd": False,
        "agent": "radarmind_fusion_teacher_agent_v0_30",
        "policy_mode": args.policy_mode,
        "model_path": args.model_path if model_worker else None,
        "model_adapter_name": args.model_adapter_name if model_worker else None,
        "model_device": args.model_device if model_worker else None,
        "model_final_state": model_worker.snapshot() if model_worker else {"status": "disabled"},
        "stream_transport": "multipart/x-mixed-replace MJPEG",
        "ego_controller": args.ego_controller,
        "cruise_speed_kph": args.cruise_speed_kph,
        "cleaned_existing_actors": cleaned_actors,
        "sync": args.sync,
        "realtime": args.realtime,
        "interrupted": interrupted,
        "elapsed_sec": round(time.time() - started_at, 2),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=7860)
    parser.add_argument("--ssh-user", default="user")
    parser.add_argument("--ssh-host", default="SERVER_IP")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--action-policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--policy-mode", choices=("hybrid", "radarmind", "safety"), default="hybrid")
    parser.add_argument("--model-path", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--model-adapter-name", default="")
    parser.add_argument("--model-adapter-path", default="")
    parser.add_argument("--model-registry", default="models/model_registry.json")
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=("bf16", "fp16", "auto"), default="bf16")
    parser.add_argument("--model-interval-sec", type=float, default=2.0)
    parser.add_argument("--model-action-ttl-sec", type=float, default=8.0)
    parser.add_argument("--model-max-new-tokens", type=int, default=128)
    parser.add_argument("--ego-controller", choices=("traffic_manager", "waypoint"), default="traffic_manager")
    parser.add_argument("--cruise-speed-kph", type=float, default=25.0)
    parser.add_argument("--monitor-speed-kph", type=float, default=18.0)
    parser.add_argument("--slow-speed-kph", type=float, default=10.0)
    parser.add_argument("--brake-speed-kph", type=float, default=4.0)
    parser.add_argument("--follow-distance-m", type=float, default=3.0)
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--cleanup-existing", action="store_true", default=True)
    parser.add_argument("--no-cleanup-existing", dest="cleanup_existing", action="store_false")
    parser.add_argument("--sync", action="store_true", default=True)
    parser.add_argument("--async-mode", dest="sync", action="store_false")
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--step-sec", type=float, default=0.05)
    parser.add_argument("--realtime", action="store_true", default=True)
    parser.add_argument("--no-realtime", dest="realtime", action="store_false")
    parser.add_argument("--save-every", type=int, default=100)

    parser.add_argument("--image-width", type=int, default=800)
    parser.add_argument("--image-height", type=int, default=450)
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--camera-x", type=float, default=1.5)
    parser.add_argument("--camera-z", type=float, default=2.4)
    parser.add_argument("--camera-pitch", type=float, default=-10.0)

    parser.add_argument("--radar-x", type=float, default=2.0)
    parser.add_argument("--radar-z", type=float, default=1.0)
    parser.add_argument("--radar-pitch", type=float, default=2.0)
    parser.add_argument("--radar-range", type=float, default=50.0)
    parser.add_argument("--radar-horizontal-fov", type=float, default=40.0)
    parser.add_argument("--radar-vertical-fov", type=float, default=4.0)
    parser.add_argument("--radar-points-per-second", type=int, default=3000)
    parser.add_argument("--corridor-half-width", type=float, default=2.4)
    parser.add_argument("--radar-ground-clearance", type=float, default=0.35)

    parser.add_argument("--slow-range", type=float, default=28.0)
    parser.add_argument("--brake-range", type=float, default=16.0)
    parser.add_argument("--emergency-range", type=float, default=8.0)
    parser.add_argument("--emergency-min-closing-speed-mps", type=float, default=0.75)
    parser.add_argument("--emergency-ttc-seconds", type=float, default=2.0)
    parser.add_argument("--route-lookahead", type=float, default=8.0)
    parser.add_argument("--steer-gain", type=float, default=1.35)
    parser.add_argument("--max-steer", type=float, default=0.65)
    parser.add_argument("--auto-reset", action="store_true", default=False)
    parser.add_argument("--no-auto-reset", dest="auto_reset", action="store_false")
    parser.add_argument("--reset-stopped-steps", type=int, default=60)
    parser.add_argument("--stopped-speed-kph", type=float, default=0.5)

    parser.add_argument("--tm-port", type=int, default=8050)
    parser.add_argument("--npc-count", type=int, default=12)
    parser.add_argument("--cyclist-count", type=int, default=5)
    parser.add_argument("--pedestrian-count", type=int, default=20)
    parser.add_argument("--pedestrian-running-ratio", type=float, default=0.1)
    parser.add_argument("--pedestrian-crossing-factor", type=float, default=0.25)
    parser.add_argument("--spawn-obstacle", action="store_true", default=False)
    parser.add_argument("--no-spawn-obstacle", dest="spawn_obstacle", action="store_false")
    parser.add_argument("--obstacle-filter", default="vehicle.audi.tt")
    parser.add_argument("--obstacle-distance", type=float, default=35.0)
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
