#!/usr/bin/env python3
"""Check the CARLA client, server and optional VLM assets without mutating CARLA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    args = parser.parse_args()

    report: dict[str, object] = {"host": args.host, "port": args.port}
    try:
        import carla

        report["carla_python"] = getattr(carla, "__version__", "installed")
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)
        world = client.get_world()
        report["server"] = "connected"
        report["map"] = world.get_map().name
    except Exception as exc:  # audit should report all failures together
        report["server"] = "unavailable"
        report["server_error"] = f"{type(exc).__name__}: {exc}"

    for name, path in (("model", args.model_path), ("adapter", args.adapter_path)):
        if path is not None:
            report[f"{name}_path"] = str(path.resolve())
            report[f"{name}_exists"] = path.exists()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["server"] != "connected":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
