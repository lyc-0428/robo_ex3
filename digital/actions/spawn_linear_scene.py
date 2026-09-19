#!/usr/bin/env python3
"""Spawn exactly three balls and three bottles on a straight line."""
import argparse
from pathlib import Path
from linear_sorting_core import make_linear_layout
from spawn_sorting_scene import spawn, wait_for_file, write_ready_file
from vision_common import TENNIS_SPAWN_HEIGHT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--controller-ready-file", type=Path, required=True)
    parser.add_argument("--scene-ready-file", type=Path, required=True)
    args = parser.parse_args()
    wait_for_file(args.controller_ready_file, 120, "linear controller")
    for item in make_linear_layout(args.seed):
        item["z"] = TENNIS_SPAWN_HEIGHT if item["class_name"] == "tennis" else 0.0
        model_root = args.model_root
        tennis_debug = model_root/"tennis_debug"
        tennis_debug_mesh = tennis_debug/"056_tennis_ball"/"textured.obj"
        if item["class_name"] == "tennis" and tennis_debug_mesh.is_file():
            model_root = tennis_debug
        spawn(item, model_root)
    write_ready_file(args.scene_ready_file)
    print("LINEAR SCENE READY | fixed cells=G1..G7 | G4=empty | tennis=3 bottle=3", flush=True)


if __name__ == "__main__":
    main()
