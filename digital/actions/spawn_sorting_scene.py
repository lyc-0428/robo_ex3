#!/usr/bin/env python3
"""Spawn two bottles and two tennis balls into four neutral slots."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

from vision_common import BOTTLE, bottle_sdf, make_slot_layout, tennis_sdf


WORLD_REMOVE_SERVICE = "/world/pick_place/remove"
DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"
DEFAULT_ANGLES = (-30.0, -10.0, 10.0, 30.0)


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
    sdf_text = (
        bottle_sdf(item["name"], model_root)
        if item["class_name"] == BOTTLE
        else tennis_sdf(item["name"], model_root)
    )
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
    finally:
        sdf_path.unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--slot-radius", type=float, default=0.350)
    parser.add_argument(
        "--slot-angles",
        type=float,
        nargs=4,
        default=DEFAULT_ANGLES,
        metavar=("A0", "A1", "A2", "A3"),
    )
    parser.add_argument("--service-timeout-ms", type=int, default=15000)
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

    layout = make_slot_layout(args.seed, args.slot_radius, args.slot_angles)
    if args.remove_existing:
        # Also remove index 5 left by the previous six-object scene.
        for index in range(6):
            remove_if_present(f"task_object_{index}", args.service_timeout_ms)
    for item in layout:
        spawn(item, args.model_root)

    # Deliberately do not expose the shuffled class map to the sorting node.
    print(
        "SORTING SCENE READY | "
        f"seed={args.seed} | objects=4 | bottles=2 | tennis=2 | "
        f"radius={args.slot_radius:.3f}"
    )


if __name__ == "__main__":
    main()
