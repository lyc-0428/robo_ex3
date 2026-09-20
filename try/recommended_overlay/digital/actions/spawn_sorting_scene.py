#!/usr/bin/env python3
"""Spawn six randomized task objects after the robot controllers are ready."""

from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from vision_common import (
    BOTTLE,
    TENNIS,
    TENNIS_SPAWN_HEIGHT,
    UNKNOWN,
    bottle_sdf,
    make_slot_layout,
    tennis_sdf,
    unknown_sdf,
)


WORLD_REMOVE_SERVICE = "/world/pick_place/remove"
DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"
DEFAULT_ANGLES = (-40.0, -24.0, -8.0, 8.0, 24.0, 40.0)
TENNIS_ONLY_ANGLES = DEFAULT_ANGLES


def make_tennis_only_layout(radius, min_separation=0.12):
    """Create six separated tennis targets for placement-only debugging."""
    radius = float(radius)
    layout = []
    for index, angle_degrees in enumerate(TENNIS_ONLY_ANGLES):
        angle = math.radians(angle_degrees)
        layout.append({
            "name": f"task_object_{index}",
            "class_name": TENNIS,
            "angle_degrees": angle_degrees,
            "x": radius * math.cos(angle),
            "y": radius * math.sin(angle),
            "z": TENNIS_SPAWN_HEIGHT,
        })
    separations = [
        math.hypot(right["x"] - left["x"], right["y"] - left["y"])
        for left, right in zip(layout, layout[1:])
    ]
    if separations and min(separations) < float(min_separation):
        raise ValueError(
            f"tennis-only separation {min(separations):.3f} m is below "
            f"the required {float(min_separation):.3f} m"
        )
    return layout


def run(command, timeout=30.0, check=True):
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}"
        )
    return result


def remove_if_present(name, timeout_ms):
    run(
        [
            "ign",
            "service",
            "-s",
            WORLD_REMOVE_SERVICE,
            "--reqtype",
            "ignition.msgs.Entity",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            str(timeout_ms),
            "--req",
            f'name: "{name}"',
        ],
        timeout=timeout_ms / 1000.0 + 5.0,
        check=False,
    )


def spawn(item, model_root):
    if item["class_name"] == BOTTLE:
        sdf_text = bottle_sdf(item["name"], model_root)
    elif item["class_name"] == UNKNOWN:
        sdf_text = unknown_sdf(item["name"])
    else:
        sdf_text = tennis_sdf(item["name"], model_root)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".sdf", delete=False
    ) as stream:
        stream.write(sdf_text)
        sdf_path = Path(stream.name)
    try:
        run(
            [
                "ros2",
                "run",
                "ros_gz_sim",
                "create",
                "-world",
                "pick_place",
                "-name",
                item["name"],
                "-file",
                str(sdf_path),
                "-x",
                f"{item['x']:.9f}",
                "-y",
                f"{item['y']:.9f}",
                "-z",
                f"{item['z']:.9f}",
            ],
            timeout=30.0,
        )
        # ros_gz_sim may acknowledge the request before Gazebo's parser has
        # finished opening the file. Keep it alive briefly for that handoff.
        time.sleep(0.5)
    finally:
        sdf_path.unlink(missing_ok=True)


def wait_for_file(path, timeout_seconds, label):
    if path is None:
        return
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.10)
    raise RuntimeError(f"{label} did not become ready within {timeout_seconds:.1f} s")


def write_ready_file(path):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("ready\n", encoding="utf-8")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--slot-radius", type=float, default=0.470)
    parser.add_argument(
        "--slot-angles",
        type=float,
        nargs="+",
        default=DEFAULT_ANGLES,
        metavar="DEG",
    )
    parser.add_argument("--minimum-separation", type=float, default=0.120)
    parser.add_argument(
        "--scenario",
        choices=("nominal", "unknown", "empty", "tennis_only"),
        default="nominal",
        help=(
            "nominal sorts six; tennis_only spawns six balls for placement "
            "debugging; unknown and empty exercise required exceptions"
        ),
    )
    parser.add_argument("--service-timeout-ms", type=int, default=15000)
    parser.add_argument("--controller-ready-file", type=Path)
    parser.add_argument("--scene-ready-file", type=Path)
    parser.add_argument("--ready-timeout-seconds", type=float, default=90.0)
    parser.add_argument(
        "--remove-existing",
        action="store_true",
        help="remove old task_object entities before spawning (live-world reuse only)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if shutil.which("ros2") is None or shutil.which("ign") is None:
        raise SystemExit("ros2 and ign must be available in PATH")
    if args.service_timeout_ms < 1000:
        raise SystemExit("service timeout must be at least 1000 ms")

    wait_for_file(
        args.controller_ready_file,
        args.ready_timeout_seconds,
        "robot controllers",
    )
    if args.scenario == "tennis_only":
        layout = make_tennis_only_layout(
            args.slot_radius,
            min_separation=args.minimum_separation,
        )
    else:
        layout = make_slot_layout(
            args.seed,
            args.slot_radius,
            args.slot_angles,
            min_separation=args.minimum_separation,
        )
    total_slots = len(layout)
    if args.scenario == "unknown":
        layout[-1] = dict(layout[-1], class_name=UNKNOWN, z=0.045)
    elif args.scenario == "empty":
        layout = layout[:-1]
    if args.remove_existing:
        for index in range(max(12, len(layout))):
            remove_if_present(f"task_object_{index}", args.service_timeout_ms)
    for item in layout:
        spawn(item, args.model_root)
    write_ready_file(args.scene_ready_file)

    # Entity names remain neutral; the sorting node still obtains classes from YOLO.
    counts = Counter(item["class_name"] for item in layout)
    print(
        "SORTING SCENE READY | "
        f"seed={args.seed} | scenario={args.scenario} | "
        f"slots={total_slots} | objects={len(layout)} | "
        f"bottles={counts['bottle']} | tennis={counts['tennis']} | "
        f"unknown={counts['unknown']} | "
        f"radius={args.slot_radius:.3f} | "
        f"minimum_separation={args.minimum_separation:.3f}"
    )


if __name__ == "__main__":
    main()
