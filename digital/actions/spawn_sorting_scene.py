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

from placement_zone import placement_distance_for_attempt
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

# ----------------------------------------------------------------------
# Purely visual Gazebo floor markers.
#
# IMPORTANT:
# These markers intentionally contain NO <collision> element.
# They are only visual guides and must never influence simulation physics.
# ----------------------------------------------------------------------
MARKER_SIDE = 0.120
MARKER_BORDER = 0.008
MARKER_HEIGHT = 0.003
MARKER_Z = MARKER_HEIGHT * 0.5

# Keep these equal to grasp_bottle_tennis.py defaults.
PLACEMENT_DISTANCE_START = 0.850
PLACEMENT_DISTANCE_DECREMENT = 0.120
PLACEMENT_DISTANCE_MINIMUM = 0.250

# Object position relative to chassis while held at the release pose.
# These reuse the grasp standoff used by the sorting controller.
BOTTLE_RELEASE_OFFSET = 0.350
TENNIS_RELEASE_OFFSET = 0.300


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


def square_marker_sdf(name, side=MARKER_SIDE, border=MARKER_BORDER):
    """Return a visual-only black square frame.

    The model contains four thin box visuals and deliberately has no
    collision geometry, inertial body, contact settings, or friction.
    """
    side = float(side)
    border = float(border)

    if side <= 0.0:
        raise ValueError("marker side must be positive")
    if border <= 0.0 or border >= side * 0.5:
        raise ValueError("marker border width is invalid")

    edge = (side - border) * 0.5

    def visual(name_suffix, x, y, size_x, size_y):
        return f"""
      <visual name="{name_suffix}">
        <pose>{x:.9f} {y:.9f} 0 0 0 0</pose>
        <geometry>
          <box>
            <size>{size_x:.9f} {size_y:.9f} {MARKER_HEIGHT:.9f}</size>
          </box>
        </geometry>
        <material>
          <ambient>0 0 0 1</ambient>
          <diffuse>0 0 0 1</diffuse>
          <specular>0 0 0 1</specular>
        </material>
      </visual>"""

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <static>true</static>
    <link name="marker_link">
{visual("top", 0.0, edge, side, border)}
{visual("bottom", 0.0, -edge, side, border)}
{visual("left", -edge, 0.0, border, side)}
{visual("right", edge, 0.0, border, side)}
    </link>
  </model>
</sdf>
"""

    # This is a safety contract, not merely a test.
    if "<collision" in sdf.lower():
        raise RuntimeError(
            f"visual marker {name} unexpectedly contains collision geometry"
        )

    return sdf


def spawn_marker(name, x, y):
    """Spawn one static visual-only marker in the Gazebo world."""
    sdf_text = square_marker_sdf(name)

    if "<collision" in sdf_text.lower():
        raise RuntimeError(
            f"refusing to spawn {name}: collision geometry detected"
        )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".sdf",
        delete=False,
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
                name,
                "-file",
                str(sdf_path),
                "-x",
                f"{float(x):.9f}",
                "-y",
                f"{float(y):.9f}",
                "-z",
                f"{MARKER_Z:.9f}",
            ],
            timeout=30.0,
        )

        # Keep the temporary SDF alive until Gazebo has consumed it.
        time.sleep(0.15)

    finally:
        sdf_path.unlink(missing_ok=True)


def make_goal_marker_layout(layout):
    """Compute six expected object release locations.

    The sorting controller always selects camera-left first.  With the robot
    starting at world (0, 0) facing +X, this is descending slot angle.

    Bottle placement:
        right side, -90 degrees

    Tennis placement:
        left side, +90 degrees

    The marker is placed at the expected OBJECT centre, not chassis centre,
    therefore the held-object standoff is added to placement travel.
    """
    ordered = sorted(
        layout,
        key=lambda item: float(item["angle_degrees"]),
        reverse=True,
    )

    goals = []

    for attempt, item in enumerate(ordered, start=1):
        class_name = item["class_name"]

        if class_name not in (BOTTLE, TENNIS):
            raise ValueError(
                f"cannot make placement marker for class {class_name!r}"
            )

        travel = placement_distance_for_attempt(
            PLACEMENT_DISTANCE_START,
            PLACEMENT_DISTANCE_DECREMENT,
            PLACEMENT_DISTANCE_MINIMUM,
            attempt,
        )

        if class_name == BOTTLE:
            # Robot right = world -Y when initial yaw is zero.
            release_offset = BOTTLE_RELEASE_OFFSET
            direction = -1.0
        else:
            # Robot left = world +Y when initial yaw is zero.
            release_offset = TENNIS_RELEASE_OFFSET
            direction = 1.0

        object_distance = travel + release_offset

        goals.append(
            {
                "name": f"goal_marker_{attempt - 1}",
                "attempt": attempt,
                "class_name": class_name,
                "source_name": item["name"],
                "x": 0.0,
                "y": direction * object_distance,
                "travel_distance": travel,
                "release_offset": release_offset,
            }
        )

    return goals


def spawn_scene_markers(layout):
    """Spawn exactly six initial and six final visual markers."""
    if len(layout) != 6:
        raise ValueError(
            f"marker layout requires exactly six objects, got {len(layout)}"
        )

    initial_markers = []

    for index, item in enumerate(layout):
        marker = {
            "name": f"initial_marker_{index}",
            "x": float(item["x"]),
            "y": float(item["y"]),
            "class_name": item["class_name"],
            "source_name": item["name"],
        }

        initial_markers.append(marker)

    goal_markers = make_goal_marker_layout(layout)

    if len(initial_markers) != 6 or len(goal_markers) != 6:
        raise RuntimeError("expected exactly 6 initial + 6 goal markers")

    for marker in initial_markers:
        spawn_marker(
            marker["name"],
            marker["x"],
            marker["y"],
        )

        print(
            "INITIAL MARKER | "
            f"name={marker['name']} | "
            f"object={marker['source_name']} | "
            f"class={marker['class_name']} | "
            f"x={marker['x']:.3f} | "
            f"y={marker['y']:.3f} | "
            "collision=false"
        )

    for marker in goal_markers:
        spawn_marker(
            marker["name"],
            marker["x"],
            marker["y"],
        )

        print(
            "GOAL MARKER | "
            f"name={marker['name']} | "
            f"attempt={marker['attempt']} | "
            f"object={marker['source_name']} | "
            f"class={marker['class_name']} | "
            f"x={marker['x']:.3f} | "
            f"y={marker['y']:.3f} | "
            f"travel={marker['travel_distance']:.3f} | "
            f"held_offset={marker['release_offset']:.3f} | "
            "collision=false"
        )

    print(
        "VISUAL MARKERS READY | "
        "initial=6 | goal=6 | total=12 | collision=false"
    )



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
    # Keep the complete six-slot scene geometry for visual markers.
    marker_layout = [dict(item) for item in layout]

    total_slots = len(layout)
    if args.scenario == "unknown":
        layout[-1] = dict(layout[-1], class_name=UNKNOWN, z=0.045)
    elif args.scenario == "empty":
        layout = layout[:-1]
    if args.remove_existing:
        for index in range(max(12, len(layout))):
            remove_if_present(
                f"task_object_{index}",
                args.service_timeout_ms,
            )

        # Visual-only marker cleanup.
        for index in range(6):
            remove_if_present(
                f"initial_marker_{index}",
                args.service_timeout_ms,
            )
            remove_if_present(
                f"goal_marker_{index}",
                args.service_timeout_ms,
            )
    for item in layout:
        spawn(item, args.model_root)

    # The normal six-object experiment displays exactly twelve floor frames:
    # six source positions and six expected release positions.
    #
    # unknown / empty are exception-testing scenarios, so do not display
    # misleading final classification targets there.
    if args.scenario in {"nominal", "tennis_only"}:
        spawn_scene_markers(marker_layout)

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
