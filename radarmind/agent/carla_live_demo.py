"""Run a real-time CARLA radar closed loop with a browser dashboard.

The demo spawns an ego vehicle, RGB camera and CARLA radar sensor. A lightweight
radar safety agent converts the newest radar detections into RadarMind actions,
the shared :class:`ActionPolicy` maps them to ``carla.VehicleControl``, and a
small HTTP server exposes the live camera, radar BEV and control state.

Run this module inside the environment where ``carla==0.9.16`` is installed.
Open the dashboard through an SSH tunnel when CARLA runs on a remote server.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from radarmind.agent.action_policy import DEFAULT_POLICY_PATH, ActionPolicy
from radarmind.agent.carla_replay import (
    choose_blueprint,
    command_to_vehicle_control,
    require_carla,
    spawn_actor,
)


DEFAULT_OUTPUT_DIR = "runs/carla_live_demo"


LIVE_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RadarMind × CARLA Live</title>
  <style>
    :root { color-scheme: dark; --bg:#07111d; --panel:#0d1b29; --line:#1e3448;
      --text:#d9e7f2; --muted:#7f9bb2; --cyan:#50d7e8; --green:#53e38c;
      --yellow:#ffd166; --red:#ff5f68; }
    * { box-sizing:border-box; }
    body { margin:0; background:radial-gradient(circle at 25% 0,#12304b,var(--bg) 38%);
      color:var(--text); font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif; }
    header { display:flex; justify-content:space-between; align-items:center; padding:18px 24px;
      border-bottom:1px solid var(--line); background:rgba(7,17,29,.82); backdrop-filter:blur(14px);
      position:sticky; top:0; z-index:2; }
    h1 { font-size:20px; margin:0; letter-spacing:.02em; }
    .live { color:var(--green); font-weight:700; }
    .dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--green);
      box-shadow:0 0 12px var(--green); margin-right:7px; }
    main { max-width:1500px; margin:auto; padding:18px; }
    .grid { display:grid; grid-template-columns:minmax(0,1.75fr) minmax(330px,.85fr); gap:16px; }
    .panel { background:linear-gradient(145deg,rgba(18,38,56,.94),rgba(9,24,38,.94));
      border:1px solid var(--line); border-radius:14px; overflow:hidden; box-shadow:0 16px 45px #0005; }
    .panel h2 { margin:0; padding:12px 15px; font-size:13px; color:var(--muted);
      text-transform:uppercase; letter-spacing:.09em; border-bottom:1px solid var(--line); }
    img { display:block; width:100%; background:#03080d; aspect-ratio:16/9; object-fit:contain; }
    #radar { aspect-ratio:1/1; }
    .cards { display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-top:16px; }
    .card { padding:14px; border:1px solid var(--line); border-radius:12px; background:var(--panel); }
    .label { color:var(--muted); font-size:12px; }
    .value { margin-top:4px; font-size:24px; font-variant-numeric:tabular-nums; font-weight:750; }
    #action { color:var(--cyan); }
    .trace { margin-top:16px; padding:13px 15px; border:1px solid var(--line);
      border-radius:12px; background:#06101a; color:#9dc4d8; white-space:pre-wrap;
      font:13px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace; min-height:76px; }
    footer { color:var(--muted); padding:16px 2px; }
    @media(max-width:900px) {
      .grid { grid-template-columns:1fr; }
      .cards { grid-template-columns:repeat(2,1fr); }
    }
  </style>
</head>
<body>
<header>
  <h1>RadarMind <span style="color:#6689a5">×</span> CARLA Live Closed Loop</h1>
  <div class="live"><span class="dot"></span><span id="connection">CONNECTING</span></div>
</header>
<main>
  <section class="grid">
    <div class="panel"><h2>CARLA RGB camera</h2><img id="camera" src="/camera.mjpg"></div>
    <div class="panel"><h2>Radar BEV · forward 50 m</h2><img id="radar" src="/radar.mjpg"></div>
  </section>
  <section class="cards">
    <div class="card"><div class="label">ACTION</div><div class="value" id="action">warming_up</div></div>
    <div class="card"><div class="label">SPEED</div><div class="value"><span id="speed">0.0</span> <small>km/h</small></div></div>
    <div class="card"><div class="label">MIN RANGE</div><div class="value"><span id="range">--</span> <small>m</small></div></div>
    <div class="card"><div class="label">THROTTLE</div><div class="value" id="throttle">0.00</div></div>
    <div class="card"><div class="label">BRAKE</div><div class="value" id="brake">0.00</div></div>
  </section>
  <div class="trace" id="trace">Waiting for CARLA sensor frames…</div>
  <footer>Frame <span id="frame">--</span> · radar points <span id="points">0</span> ·
    state refresh <span id="updated">--</span></footer>
</main>
<script>
  const byId = id => document.getElementById(id);
  async function refresh() {
    try {
      const response = await fetch('/state.json?t=' + Date.now(), {cache:'no-store'});
      const state = await response.json();
      byId('action').textContent = state.action || 'monitor';
      byId('speed').textContent = Number(state.speed_kph || 0).toFixed(1);
      byId('range').textContent = state.radar && state.radar.min_depth_m != null
        ? Number(state.radar.min_depth_m).toFixed(1) : '--';
      byId('throttle').textContent = Number(state.control && state.control.throttle || 0).toFixed(2);
      byId('brake').textContent = Number(state.control && state.control.brake || 0).toFixed(2);
      byId('frame').textContent = state.carla_frame == null ? '--' : state.carla_frame;
      byId('points').textContent = state.radar ? state.radar.num_points : 0;
      byId('trace').textContent = state.reason || 'Radar agent is monitoring.';
      byId('updated').textContent = new Date().toLocaleTimeString();
      byId('connection').textContent = 'LIVE';
    } catch (error) {
      byId('connection').textContent = 'RECONNECTING';
    }
  }
  refresh();
  setInterval(refresh, 250);
</script>
</body>
</html>
"""


def jpeg_bytes(image: Image.Image, quality: int = 82) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False)
    return buffer.getvalue()


def placeholder_image(size: tuple[int, int], title: str) -> bytes:
    image = Image.new("RGB", size, "#06101a")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((20, 20), title, fill="#7f9bb2", font=font)
    return jpeg_bytes(image)


def carla_image_to_pil(image: Any) -> Image.Image:
    return Image.frombytes(
        "RGBA",
        (image.width, image.height),
        bytes(image.raw_data),
        "raw",
        "BGRA",
    ).convert("RGB")


@dataclass
class LiveState:
    camera_jpeg: bytes
    radar_jpeg: bytes
    state: dict[str, Any] = field(default_factory=dict)
    radar_points: list[dict[str, float]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update_camera(self, image: Any) -> None:
        frame = carla_image_to_pil(image)
        payload = jpeg_bytes(frame, quality=75)
        with self.lock:
            self.camera_jpeg = payload

    def update_radar(self, measurement: Any, max_range: float) -> None:
        points = [
            {
                "depth": float(detection.depth),
                "azimuth": float(detection.azimuth),
                "altitude": float(detection.altitude),
                "velocity": float(detection.velocity),
            }
            for detection in measurement
        ]
        image = render_radar_bev(points, max_range=max_range)
        with self.lock:
            self.radar_points = points
            self.radar_jpeg = jpeg_bytes(image)

    def snapshot(self) -> tuple[bytes, bytes, dict[str, Any], list[dict[str, float]]]:
        with self.lock:
            return (
                self.camera_jpeg,
                self.radar_jpeg,
                dict(self.state),
                list(self.radar_points),
            )


def render_radar_bev(
    points: list[dict[str, float]],
    max_range: float,
    size: int = 600,
) -> Image.Image:
    image = Image.new("RGB", (size, size), "#04101a")
    draw = ImageDraw.Draw(image)
    center_x = size // 2
    origin_y = size - 30
    usable = size - 55

    for distance in (10, 20, 30, 40, int(max_range)):
        if distance > max_range:
            continue
        radius = distance / max_range * usable
        box = (center_x - radius, origin_y - radius, center_x + radius, origin_y + radius)
        draw.arc(box, 200, 340, fill="#1a4257", width=1)
        draw.text((center_x + 5, origin_y - radius - 13), f"{distance}m", fill="#55788d")

    draw.line((center_x, origin_y, center_x, 20), fill="#1f5c73", width=1)
    draw.line((center_x, origin_y, 75, 90), fill="#153b4d", width=1)
    draw.line((center_x, origin_y, size - 75, 90), fill="#153b4d", width=1)
    draw.polygon(
        [(center_x, origin_y - 13), (center_x - 8, origin_y + 6), (center_x + 8, origin_y + 6)],
        fill="#50d7e8",
    )

    for point in points:
        depth = point["depth"]
        if depth <= 0 or depth > max_range:
            continue
        lateral = depth * math.sin(point["azimuth"])
        forward = depth * math.cos(point["azimuth"])
        px = center_x + lateral / max_range * usable
        py = origin_y - forward / max_range * usable
        velocity = point["velocity"]
        color = "#ff5f68" if velocity < -1.0 else "#ffd166" if abs(velocity) <= 1.0 else "#53e38c"
        radius = 4 if depth < 20 else 3
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)

    draw.text((14, 12), f"DETECTIONS {len(points)}", fill="#8eb0c3")
    draw.text((14, 30), "red: closing  yellow: static  green: receding", fill="#55788d")
    return image


def radar_summary(points: list[dict[str, float]]) -> dict[str, Any]:
    if not points:
        return {
            "num_points": 0,
            "min_depth_m": None,
            "nearest_velocity_mps": None,
            "approaching_points": 0,
        }
    nearest = min(points, key=lambda point: point["depth"])
    return {
        "num_points": len(points),
        "min_depth_m": round(nearest["depth"], 3),
        "nearest_velocity_mps": round(nearest["velocity"], 3),
        "approaching_points": sum(point["velocity"] < -1.0 for point in points),
    }


def radar_safety_action(
    summary: dict[str, Any],
    slow_range: float,
    brake_range: float,
    emergency_range: float,
) -> dict[str, str]:
    depth = summary.get("min_depth_m")
    if depth is None:
        return {
            "type": "monitor",
            "reason": "No forward radar return yet; hold a conservative command while sensors warm up.",
        }
    if depth <= emergency_range:
        return {
            "type": "emergency_brake",
            "reason": f"Nearest radar return is {depth:.1f} m: emergency braking threshold reached.",
        }
    if depth <= brake_range:
        return {
            "type": "brake",
            "reason": f"Nearest radar return is {depth:.1f} m: braking to preserve headway.",
        }
    if depth <= slow_range:
        return {
            "type": "slow_down",
            "reason": f"Nearest radar return is {depth:.1f} m: slowing down before the target gets critical.",
        }
    return {
        "type": "keep_speed",
        "reason": f"Nearest radar return is {depth:.1f} m: forward corridor is currently safe.",
    }


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], live_state: LiveState):
        self.live_state = live_state
        super().__init__(address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(LIVE_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/camera.mjpg":
            self._stream_mjpeg("camera")
        elif path == "/radar.mjpg":
            self._stream_mjpeg("radar")
        elif path == "/camera.jpg":
            camera, _, _, _ = self.server.live_state.snapshot()
            self._send(camera, "image/jpeg")
        elif path == "/radar.jpg":
            _, radar, _, _ = self.server.live_state.snapshot()
            self._send(radar, "image/jpeg")
        elif path in {"/state.json", "/health"}:
            _, _, state, _ = self.server.live_state.snapshot()
            self._send(
                (json.dumps(state, ensure_ascii=False) + "\n").encode("utf-8"),
                "application/json; charset=utf-8",
            )
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _stream_mjpeg(self, stream: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        previous: bytes | None = None
        try:
            while True:
                camera, radar, _, _ = self.server.live_state.snapshot()
                payload = camera if stream == "camera" else radar
                if payload != previous:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(payload)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    previous = payload
                time.sleep(0.04)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return

    def _send(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def start_dashboard(host: str, port: int, live_state: LiveState) -> DashboardServer:
    server = DashboardServer((host, port), live_state)
    thread = threading.Thread(target=server.serve_forever, name="radarmind-dashboard", daemon=True)
    thread.start()
    return server


def cleanup_live_actors(world: Any) -> int:
    removed = 0
    for actor in world.get_actors():
        try:
            role_name = actor.attributes.get("role_name", "")
        except RuntimeError:
            continue
        if role_name in {"radarmind_ego", "radarmind_obstacle"}:
            actor.destroy()
            removed += 1
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

    road_map = world.get_map()
    origin_waypoint = road_map.get_waypoint(
        origin.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    candidate_distances = [distance, distance + 5.0, distance - 5.0, distance + 10.0, distance - 10.0]
    for candidate_distance in candidate_distances:
        if candidate_distance <= 5.0:
            continue
        waypoints = origin_waypoint.next(candidate_distance)
        for waypoint in waypoints:
            transform = waypoint.transform
            transform.location.z += 0.25
            actor = world.try_spawn_actor(blueprint, transform)
            if actor is not None:
                actor.set_simulate_physics(False)
                return actor

    # A final geometric fallback keeps the demo usable on maps whose lane graph
    # has no forward waypoint from the selected spawn position.
    yaw = math.radians(origin.rotation.yaw)
    transform = carla.Transform(
        carla.Location(
            x=origin.location.x + distance * math.cos(yaw),
            y=origin.location.y + distance * math.sin(yaw),
            z=origin.location.z + 0.25,
        ),
        origin.rotation,
    )
    actor = world.try_spawn_actor(blueprint, transform)
    if actor is not None:
        actor.set_simulate_physics(False)
    return actor


def speed_kph(vehicle: Any) -> float:
    velocity = vehicle.get_velocity()
    return 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


def run(args: argparse.Namespace) -> dict[str, Any]:
    carla = require_carla()
    output_dir = Path(args.output_dir)
    snapshots_dir = output_dir / "snapshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "live_trace.jsonl"
    report_path = output_dir / "live_demo.report.json"

    live_state = LiveState(
        camera_jpeg=placeholder_image((args.image_width, args.image_height), "Waiting for RGB camera"),
        radar_jpeg=placeholder_image((600, 600), "Waiting for radar detections"),
        state={"status": "starting", "action": "monitor"},
    )
    dashboard = start_dashboard(args.web_host, args.web_port, live_state)

    rng = random.Random(args.seed)
    policy = ActionPolicy.from_path(args.action_policy)
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    original_settings = world.get_settings()
    vehicle = None
    camera = None
    radar = None
    obstacle = None
    cleaned_actors = 0
    records = 0
    snapshots_saved = 0
    started_at = time.time()
    interrupted = False

    try:
        if args.sync:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = args.fixed_delta_seconds
            world.apply_settings(settings)

        if args.cleanup_existing:
            cleaned_actors = cleanup_live_actors(world)
            if args.sync:
                world.tick()

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("CARLA map has no spawn points")
        spawn_transform = spawn_points[args.spawn_index % len(spawn_points)]
        vehicle_bp = choose_blueprint(world, args.vehicle_filter, rng)
        vehicle = spawn_actor(world, vehicle_bp, spawn_transform, retries=len(spawn_points))

        if args.spawn_obstacle:
            obstacle = spawn_lead_vehicle(
                carla,
                world,
                vehicle.get_transform(),
                pattern=args.obstacle_filter,
                distance=args.obstacle_distance,
                rng=rng,
            )

        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(args.image_width))
        camera_bp.set_attribute("image_size_y", str(args.image_height))
        camera_bp.set_attribute("fov", str(args.camera_fov))
        camera_transform = carla.Transform(
            carla.Location(x=args.camera_x, y=0.0, z=args.camera_z),
            carla.Rotation(pitch=args.camera_pitch),
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        camera.listen(live_state.update_camera)

        radar_bp = world.get_blueprint_library().find("sensor.other.radar")
        radar_bp.set_attribute("horizontal_fov", str(args.radar_horizontal_fov))
        radar_bp.set_attribute("vertical_fov", str(args.radar_vertical_fov))
        radar_bp.set_attribute("range", str(args.radar_range))
        radar_bp.set_attribute("points_per_second", str(args.radar_points_per_second))
        radar_bp.set_attribute("sensor_tick", str(args.fixed_delta_seconds))
        radar_transform = carla.Transform(
            carla.Location(x=args.radar_x, y=0.0, z=args.radar_z),
            carla.Rotation(pitch=args.radar_pitch),
        )
        radar = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)
        radar.listen(lambda measurement: live_state.update_radar(measurement, args.radar_range))

        print(
            json.dumps(
                {
                    "dashboard": f"http://{args.web_host}:{args.web_port}",
                    "ssh_tunnel": (
                        f"ssh -L {args.web_port}:127.0.0.1:{args.web_port} "
                        f"{args.ssh_user}@{args.ssh_host}"
                    ),
                    "output_dir": str(output_dir),
                    "mode": "run until Ctrl+C" if args.max_steps == 0 else f"{args.max_steps} steps",
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )

        with trace_path.open("w", encoding="utf-8") as trace_handle:
            step_idx = 0
            while args.max_steps == 0 or step_idx < args.max_steps:
                step_idx += 1
                frame_id = world.tick() if args.sync else None
                if not args.sync:
                    time.sleep(args.step_sec)

                camera_jpeg, radar_jpeg, _, points = live_state.snapshot()
                summary = radar_summary(points)
                action = radar_safety_action(
                    summary,
                    slow_range=args.slow_range,
                    brake_range=args.brake_range,
                    emergency_range=args.emergency_range,
                )
                policy_command = policy.command_for(action).to_dict()
                control = command_to_vehicle_control(carla, policy_command)
                vehicle.apply_control(control)

                current_speed = speed_kph(vehicle)
                state = {
                    "status": "live",
                    "step": step_idx,
                    "carla_frame": int(frame_id) if frame_id is not None else None,
                    "timestamp": time.time(),
                    "map": world.get_map().name,
                    "action": action["type"],
                    "reason": action["reason"],
                    "radar": summary,
                    "speed_kph": round(current_speed, 3),
                    "control": {
                        "throttle": round(float(control.throttle), 3),
                        "brake": round(float(control.brake), 3),
                        "steer": round(float(control.steer), 3),
                    },
                    "agent": "radar_rule_safety_agent_v0_26",
                    "obstacle_spawned": obstacle is not None,
                }
                with live_state.lock:
                    live_state.state = state

                trace_handle.write(json.dumps(state, ensure_ascii=False) + "\n")
                trace_handle.flush()
                records += 1

                if args.save_every > 0 and step_idx % args.save_every == 0:
                    snapshots_dir.mkdir(parents=True, exist_ok=True)
                    (snapshots_dir / f"camera_{step_idx:06d}.jpg").write_bytes(camera_jpeg)
                    (snapshots_dir / f"radar_{step_idx:06d}.jpg").write_bytes(radar_jpeg)
                    snapshots_saved += 2
    except KeyboardInterrupt:
        interrupted = True
    finally:
        dashboard.shutdown()
        dashboard.server_close()
        for actor in (radar, camera):
            if actor is not None:
                actor.stop()
                actor.destroy()
        for actor in (obstacle, vehicle):
            if actor is not None:
                actor.destroy()
        if args.sync:
            world.apply_settings(original_settings)

    report = {
        "output_dir": str(output_dir),
        "trace_path": str(trace_path),
        "report_path": str(report_path),
        "dashboard_url": f"http://{args.web_host}:{args.web_port}",
        "host": args.host,
        "port": args.port,
        "map": world.get_map().name,
        "records": records,
        "snapshots_saved": snapshots_saved,
        "vehicle_filter": args.vehicle_filter,
        "spawn_index": args.spawn_index,
        "obstacle_spawned": obstacle is not None,
        "cleaned_existing_actors": cleaned_actors,
        "action_policy": args.action_policy,
        "agent": "radar_rule_safety_agent_v0_26",
        "sync": args.sync,
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
    parser.add_argument("--ssh-host", default="<server-ip>")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until Ctrl+C")
    parser.add_argument("--action-policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cleanup-existing", action="store_true", default=True)
    parser.add_argument("--no-cleanup-existing", dest="cleanup_existing", action="store_false")
    parser.add_argument("--sync", action="store_true", default=True)
    parser.add_argument("--async-mode", dest="sync", action="store_false")
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--step-sec", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=100)

    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--camera-x", type=float, default=1.5)
    parser.add_argument("--camera-z", type=float, default=2.4)
    parser.add_argument("--camera-pitch", type=float, default=-10.0)

    parser.add_argument("--radar-x", type=float, default=2.0)
    parser.add_argument("--radar-z", type=float, default=1.0)
    parser.add_argument("--radar-pitch", type=float, default=0.0)
    parser.add_argument("--radar-range", type=float, default=50.0)
    parser.add_argument("--radar-horizontal-fov", type=float, default=40.0)
    parser.add_argument("--radar-vertical-fov", type=float, default=10.0)
    parser.add_argument("--radar-points-per-second", type=int, default=3000)

    parser.add_argument("--slow-range", type=float, default=28.0)
    parser.add_argument("--brake-range", type=float, default=16.0)
    parser.add_argument("--emergency-range", type=float, default=8.0)
    parser.add_argument("--spawn-obstacle", action="store_true", default=True)
    parser.add_argument("--no-spawn-obstacle", dest="spawn_obstacle", action="store_false")
    parser.add_argument("--obstacle-filter", default="vehicle.audi.tt")
    parser.add_argument("--obstacle-distance", type=float, default=35.0)
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
