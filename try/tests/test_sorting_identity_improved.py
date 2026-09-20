from __future__ import annotations

import math
from pathlib import Path
import sys
import time
import unittest

import numpy as np


ACTIONS = (
    Path(__file__).resolve().parents[1]
    / "recommended_overlay" / "digital" / "actions"
)
sys.path.insert(0, str(ACTIONS))

from sorting_identity import SortingIdentity  # noqa: E402


class ImprovedIdentityTests(unittest.TestCase):
    def make_identity(self):
        identity = SortingIdentity.__new__(SortingIdentity)
        identity.completed = set()
        identity.updated = time.monotonic()
        identity._lock = None
        identity.camera = lambda _state: np.eye(4)
        identity.poses = {
            "robomaster_ep_core": {
                "position": {"x": 0.1, "y": -0.2},
                "orientation": {
                    "z": math.sin(0.3),
                    "w": math.cos(0.3),
                },
            },
            "task_object_0": {"position": {"x": 1.0, "y": 0.3}},
            "task_object_1": {"position": {"x": 1.0, "y": -0.3}},
        }
        return identity

    def test_pose2d_reports_position_and_quaternion_yaw(self):
        pose = self.make_identity().pose2d()
        self.assertAlmostEqual(pose.x, 0.1)
        self.assertAlmostEqual(pose.y, -0.2)
        self.assertAlmostEqual(pose.yaw, 0.6)

    def test_active_objects_exclude_completed(self):
        identity = self.make_identity()
        identity.completed.add("task_object_0")
        self.assertEqual(identity.active_task_objects(), ["task_object_1"])

    def test_class_limit_is_configurable_not_hardcoded_to_two(self):
        identity = self.make_identity()
        boxes = [
            {
                "class_name": "tennis",
                "bbox": {"x1": 209, "x2": 239, "y1": 225, "y2": 255},
            },
            {
                "class_name": "tennis",
                "bbox": {"x1": 401, "x2": 431, "y1": 225, "y2": 255},
            },
        ]
        accepted = identity.associate(
            boxes,
            None,
            (320, 320, 320, 240),
            counts={"tennis": 2},
            class_limits={"tennis": 3},
        )
        self.assertEqual(len(accepted), 2)
        rejected = identity.associate(
            boxes,
            None,
            (320, 320, 320, 240),
            counts={"tennis": 3},
            class_limits={"tennis": 3},
        )
        self.assertEqual(rejected, [])


if __name__ == "__main__":
    unittest.main()

