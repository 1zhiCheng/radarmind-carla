"""Replay RadarMind actions in a running CARLA server.

This is the first real-CARLA bridge for RadarMind. It reads a RadarMind
``trace.jsonl`` or ``action_timeline.jsonl``, spawns an ego vehicle and an RGB
camera, applies CARLA ``VehicleControl`` commands from the RadarMind action
policy, and saves camera frames plus an execution trace.

Run this inside the conda environment where ``carla==0.9.16`` is installed.
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import time
from pathlib import Path
from typing import Any, Iterable

from radarmind.agent.action_policy import DEFAULT_POLICY_PATH, ActionPolicy
from radarmind.evaluation.action_timeline import get_action, read_jsonl


DEFAULT_INPUT_TRACE = "examples/action_trace.jsonl"
DEFAULT_OUTPUT_DIR = "runs/carla_replay"


def require_carla() -> Any:
    try:
        import carla
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: carla. Activate the environment where you installed it, e.g. "
            "`conda activate radargym-rl`, or run `python3 -m pip install carla==0.9.16`."
        ) from exc
    return carla


def command_to_vehicle_control(carla: Any, command: dict[str, Any]) -> Any:
    return carla.VehicleControl(
        throttle=float(command.get("throttle", 0.0) or 0.0),
        brake=float(command.get("brake", 0.0) or 0.0),
        steer=float(command.get("steer", 0.0) or 0.0),
        reverse=False,
        hand_brake=False,
        manual_gear_shift=False,
    )


def save_rgb_image(image: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save_to_disk(str(output_path))


def choose_blueprint(world: Any, pattern: str, rng: random.Random) -> Any:
    blueprints = list(world.get_blueprint_library().filter(pattern))
    if not blueprints:
        raise RuntimeError(f"No CARLA blueprint matched {pattern!r}")
    blueprint = rng.choice(blueprints)
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "radarmind_ego")
    return blueprint


def cleanup_radarmind_actors(world: Any) -> int:
    removed = 0
    for actor in world.get_actors():
        try:
            role_name = actor.attributes.get("role_name", "")
        except RuntimeError:
            continue
        if role_name == "radarmind_ego":
            actor.destroy()
            removed += 1
    return removed


def spawn_actor(world: Any, blueprint: Any, transform: Any, retries: int = 20) -> Any:
    spawn_points = world.get_map().get_spawn_points()
    candidates = []
    if transform is not None:
        candidates.append(transform)
    candidates.extend(spawn_points)
    last_error: Exception | None = None
    tried = 0
    for candidate in candidates:
        if tried >= retries:
            break
        tried += 1
        try:
            actor = world.try_spawn_actor(blueprint, candidate)
            if actor is not None:
                return actor
        except RuntimeError as exc:
            last_error = exc
    if last_error:
        raise RuntimeError(f"Failed to spawn actor after {tried} attempts: {last_error}") from last_error
    raise RuntimeError(f"Failed to spawn actor after {tried} attempts: all tried spawn points were occupied")


def run(args: argparse.Namespace) -> dict[str, Any]:
    carla = require_carla()
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "camera_frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.input_trace)
    if args.max_steps > 0:
        rows = rows[: args.max_steps]
    if not rows:
        raise ValueError(f"No rows found in {args.input_trace}")

    policy = ActionPolicy.from_path(args.action_policy)
    rng = random.Random(args.seed)
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    original_settings = world.get_settings()

    vehicle = None
    camera = None
    image_queue: queue.Queue[Any] = queue.Queue()
    cleaned_actors = 0
    replay_rows: list[dict[str, Any]] = []
    started_at = time.time()

    try:
        if args.sync:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = args.fixed_delta_seconds
            world.apply_settings(settings)

        if args.cleanup_existing:
            cleaned_actors = cleanup_radarmind_actors(world)
            if args.sync:
                world.tick()

        vehicle_bp = choose_blueprint(world, args.vehicle_filter, rng)
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("CARLA map has no spawn points")
        spawn_transform = spawn_points[args.spawn_index % len(spawn_points)]
        vehicle = spawn_actor(world, vehicle_bp, spawn_transform, retries=len(spawn_points))
        # Do not enable Traffic Manager; this replay drives the vehicle directly
        # with VehicleControl. On shared servers, even set_autopilot(False) can
        # touch a Traffic Manager RPC port and fail with a bind error.

        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(args.image_width))
        camera_bp.set_attribute("image_size_y", str(args.image_height))
        camera_bp.set_attribute("fov", str(args.camera_fov))
        camera_transform = carla.Transform(
            carla.Location(x=args.camera_x, y=args.camera_y, z=args.camera_z),
            carla.Rotation(pitch=args.camera_pitch, yaw=args.camera_yaw, roll=args.camera_roll),
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        camera.listen(image_queue.put)

        if args.sync:
            world.tick()
        else:
            time.sleep(args.warmup_sec)

        for step_idx, row in enumerate(rows, start=1):
            action = get_action(row)
            policy_command = policy.command_for(action).to_dict()
            control = command_to_vehicle_control(carla, policy_command)
            vehicle.apply_control(control)

            frame_id = None
            if args.sync:
                frame_id = world.tick()
            else:
                time.sleep(args.step_sec)

            image_path = None
            try:
                image = image_queue.get(timeout=args.image_timeout)
                image_path = frames_dir / f"{step_idx:06d}.png"
                save_rgb_image(image, image_path)
            except queue.Empty:
                pass

            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            replay_rows.append(
                {
                    "idx": step_idx,
                    "source_sample_id": row.get("sample_id"),
                    "source_frame_id": row.get("frame_id"),
                    "carla_frame": int(frame_id) if frame_id is not None else None,
                    "action": action,
                    "policy_command": policy_command,
                    "vehicle_control": {
                        "throttle": control.throttle,
                        "brake": control.brake,
                        "steer": control.steer,
                    },
                    "vehicle_transform": {
                        "x": transform.location.x,
                        "y": transform.location.y,
                        "z": transform.location.z,
                        "yaw": transform.rotation.yaw,
                    },
                    "vehicle_velocity": {
                        "x": velocity.x,
                        "y": velocity.y,
                        "z": velocity.z,
                    },
                    "camera_frame_path": str(image_path) if image_path else None,
                }
            )

    finally:
        if camera is not None:
            camera.stop()
            camera.destroy()
        if vehicle is not None:
            vehicle.destroy()
        if args.sync:
            world.apply_settings(original_settings)

    trace_path = output_dir / "carla_replay_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for row in replay_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "input_trace": args.input_trace,
        "output_dir": str(output_dir),
        "trace_path": str(trace_path),
        "camera_frames_dir": str(frames_dir),
        "host": args.host,
        "port": args.port,
        "map": world.get_map().name,
        "records": len(replay_rows),
        "images_saved": sum(1 for row in replay_rows if row["camera_frame_path"]),
        "vehicle_filter": args.vehicle_filter,
        "spawn_index": args.spawn_index,
        "cleaned_existing_actors": cleaned_actors,
        "action_policy": args.action_policy,
        "sync": args.sync,
        "elapsed_sec": round(time.time() - started_at, 2),
    }
    (output_dir / "carla_replay.report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-trace", default=DEFAULT_INPUT_TRACE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--action-policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cleanup-existing", action="store_true", default=True)
    parser.add_argument("--no-cleanup-existing", dest="cleanup_existing", action="store_false")
    parser.add_argument("--sync", action="store_true", default=True)
    parser.add_argument("--async-mode", dest="sync", action="store_false")
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--step-sec", type=float, default=0.1)
    parser.add_argument("--warmup-sec", type=float, default=1.0)
    parser.add_argument("--image-timeout", type=float, default=2.0)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--camera-x", type=float, default=1.5)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--camera-z", type=float, default=2.4)
    parser.add_argument("--camera-pitch", type=float, default=-10.0)
    parser.add_argument("--camera-yaw", type=float, default=0.0)
    parser.add_argument("--camera-roll", type=float, default=0.0)
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
