from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET


ACTIONS = Path(__file__).resolve().parents[1] / "actions"
sys.path.insert(0, str(ACTIONS))

from vision_common import (  # noqa: E402
    BOTTLE,
    TENNIS,
    angle_to_steps,
    bottle_sdf,
    canonical_class,
    horizontal_angle_radians,
    make_slot_layout,
    placement_step,
    stable_center_detection,
    tennis_sdf,
    validate_model_names,
)


def detection(label, center_x, confidence=0.9):
    return {
        "class_name": label,
        "confidence": confidence,
        "bbox": {
            "x1": center_x - 10,
            "y1": 100,
            "x2": center_x + 10,
            "y2": 140,
        },
    }


class VisionCommonTests(unittest.TestCase):
    def test_class_aliases(self):
        self.assertEqual(canonical_class("water_bottle"), BOTTLE)
        self.assertEqual(canonical_class("Tennis Ball"), TENNIS)
        self.assertEqual(canonical_class("sports-ball"), TENNIS)
        self.assertIsNone(canonical_class("person"))

    def test_model_class_validation(self):
        self.assertEqual(
            validate_model_names({0: "bottle", 1: "tennis_ball"}),
            {0: BOTTLE, 1: TENNIS},
        )
        with self.assertRaisesRegex(ValueError, "actual classes"):
            validate_model_names({0: "bottle", 1: "person"})

    def test_three_of_five_stability(self):
        frames = [
            [detection("bottle", 315)],
            [detection("bottle", 321)],
            [],
            [detection("bottle", 318)],
            [detection("tennis_ball", 400)],
        ]
        result = stable_center_detection(frames, 320, required_votes=3)
        self.assertIsNotNone(result)
        self.assertEqual(result["class_name"], BOTTLE)
        self.assertEqual(result["stable_votes"], 3)

    def test_unstable_or_unknown_is_rejected(self):
        frames = [
            [detection("bottle", 100)],
            [detection("bottle", 260)],
            [detection("person", 320)],
            [],
            [],
        ]
        self.assertIsNone(
            stable_center_detection(
                frames,
                320,
                required_votes=3,
                max_center_spread_px=90,
            )
        )

    def test_pixel_angle_and_step_direction(self):
        right = horizontal_angle_radians(detection("bottle", 420), 320, 320)
        left = horizontal_angle_radians(detection("bottle", 220), 320, 320)
        self.assertGreater(right, 0)
        self.assertLess(left, 0)
        self.assertEqual(angle_to_steps(math.pi, 334.225), 1050)

    def test_placement_sequence_clamps(self):
        values = [2350, 2500, 2650]
        self.assertEqual(placement_step(values, 0), 2350)
        self.assertEqual(placement_step(values, 2), 2650)
        self.assertEqual(placement_step(values, 20), 2650)

    def test_random_layout_has_neutral_names_and_requested_counts(self):
        angles = [-30.0, -10.0, 10.0, 30.0]
        first = make_slot_layout(21, 0.350, angles)
        second = make_slot_layout(21, 0.350, angles)
        self.assertEqual(first, second)
        self.assertEqual(sum(item["class_name"] == BOTTLE for item in first), 2)
        self.assertEqual(sum(item["class_name"] == TENNIS for item in first), 2)
        self.assertTrue(
            all(item["z"] == 0.0 for item in first if item["class_name"] == BOTTLE)
        )
        self.assertTrue(
            all(
                item["z"] == 0.0335
                for item in first
                if item["class_name"] == TENNIS
            )
        )
        self.assertEqual(
            [item["name"] for item in first],
            [f"task_object_{index}" for index in range(4)],
        )
        separations = [
            math.hypot(right["x"] - left["x"], right["y"] - left["y"])
            for left, right in zip(first, first[1:])
        ]
        self.assertGreater(min(separations), 0.12)

    def test_generated_object_sdf_is_valid_xml(self):
        model_root = Path(__file__).resolve().parents[1] / "models"
        bottle = bottle_sdf("task_object_0", model_root)
        bottle_xml = ET.fromstring(bottle)
        self.assertEqual(bottle_xml.find("model").attrib["name"], "task_object_0")
        self.assertIn(
            "model://water_bottle_02/meshes/bottle_body.obj", bottle
        )
        self.assertIn(
            "model://water_bottle_02/meshes/bottle_cap.obj", bottle
        )
        for visual_name in ("bottle_body_visual", "bottle_cap_visual"):
            pose = bottle_xml.find(
                f".//visual[@name='{visual_name}']/pose"
            )
            self.assertIsNotNone(pose)
            self.assertEqual(pose.text.strip(), "0 0 -0.105 0 0 0")
        self.assertNotIn("WaterBottle_fortress.obj", bottle)
        tennis = ET.fromstring(tennis_sdf("task_object_1", model_root))
        plugin = tennis.find("./model/plugin")
        self.assertIsNotNone(plugin)
        self.assertEqual(
            plugin.get("filename"),
            "ignition-gazebo-detachable-joint-system",
        )
        self.assertEqual(plugin.findtext("parent_link"), "object_link")
        self.assertEqual(plugin.findtext("child_model"), "ground")
        self.assertEqual(plugin.findtext("child_link"), "ground_link")
        self.assertEqual(
            plugin.findtext("detach_topic"),
            "/sorting/task_object_1/ground_lock/detach",
        )
        self.assertEqual(
            plugin.findtext("attach_topic"),
            "/sorting/task_object_1/ground_lock/attach",
        )


if __name__ == "__main__":
    unittest.main()
