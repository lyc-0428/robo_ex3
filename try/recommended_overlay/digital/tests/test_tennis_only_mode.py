from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


DIGITAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIGITAL / "actions"))

from spawn_sorting_scene import make_tennis_only_layout  # noqa: E402


class TennisOnlyModeTests(unittest.TestCase):
    def test_scene_contains_six_separated_tennis_balls(self):
        layout = make_tennis_only_layout(0.470, min_separation=0.120)
        self.assertEqual(len(layout), 6)
        self.assertEqual({item["class_name"] for item in layout}, {"tennis"})
        self.assertTrue(all(math.isclose(item["z"], 0.027) for item in layout))
        distances = [
            math.hypot(right["x"] - left["x"], right["y"] - left["y"])
            for left, right in zip(layout, layout[1:])
        ]
        self.assertGreaterEqual(min(distances), 0.120)

    def test_launch_passes_scene_mode_to_controller(self):
        launch = (
            DIGITAL
            / "src/robomaster_pick_place_sim/launch/vision_sorting_improved.launch.py"
        ).read_text(encoding="utf-8")
        self.assertIn('["scene_mode:=", scenario]', launch)

    def test_release_is_guarded_by_completed_placement_motion(self):
        action = (DIGITAL / "actions/grasp_bottle_tennis.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('label == "下降到放置位" and not self._placement_ready', action)
        self.assertIn("PLACEMENT CORRIDOR ARRIVAL VERIFIED", action)

    def test_placement_stops_without_endpoint_turn_then_reverses(self):
        action = (DIGITAL / "actions/grasp_bottle_tennis.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("SET EXACT 90 DEG HEADING AT", action)
        self.assertIn("PLACEMENT CORRIDOR ARRIVAL VERIFIED", action)
        self.assertIn("DIRECT REVERSE AFTER RELEASE", action)

    def test_grasp_approach_locks_visual_heading_and_drives_straight(self):
        action = (DIGITAL / "actions/grasp_bottle_tennis.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("locked_heading = current.yaw", action)
        self.assertIn("STRAIGHT GRASP CORRIDOR LOCKED", action)
        self.assertIn('control_mode="straight"', action)
        self.assertNotIn("FACE LOCKED OBJECT", action)

    def test_missed_grasp_returns_and_reacquires_without_counting(self):
        action = (DIGITAL / "actions/grasp_bottle_tennis.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("RECOVER MISSED GRASP", action)
        self.assertIn("ROUND RETRY", action)
        self.assertIn("returned_to_origin=true | reacquire=true", action)
        self.assertIn("retry round with fresh visual acquisition", action)
        self.assertIn("self._locked_object_id = None", action)
        self.assertIn("success_count_unchanged=true", action)

    def test_real_object_lift_is_required_for_tennis_grasp(self):
        action = (DIGITAL / "actions/grasp_bottle_tennis.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("REAL GRASP VERIFIED", action)
        self.assertIn("MISSED GRASP DETECTED", action)
        self.assertIn("raise MissedGraspError", action)

    def test_small_cycle_residual_cannot_trigger_a_large_search_turn(self):
        action = (DIGITAL / "actions/grasp_bottle_tennis.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("cycle_return_tolerance_m", action)
        self.assertIn("CONFIRM RETREAT WITHIN CYCLE CENTER REGION", action)
        self.assertIn("DIRECT CLASSIFICATION TURN", action)
        self.assertNotIn("RESTORE CYCLE HEADING WITH OBJECT", action)

    def test_large_markers_cover_every_scheduled_chassis_distance(self):
        import xml.etree.ElementTree as ET

        world = ET.parse(
            DIGITAL / "src/robomaster_pick_place_sim/worlds/pick_place.sdf"
        )
        for name, sign in (("tennis_zone_marker", 1.0), ("bottle_zone_marker", -1.0)):
            model = world.find(f".//model[@name='{name}']")
            pose_y = float(model.findtext("pose").split()[1])
            size_x, size_y, _ = map(
                float, model.findtext(".//visual/geometry/box/size").split()
            )
            self.assertEqual((size_x, size_y), (1.20, 0.80))
            self.assertAlmostEqual(pose_y, sign * 0.60)
            lower = abs(pose_y) - size_y * 0.5
            upper = abs(pose_y) + size_y * 0.5
            self.assertLessEqual(lower, 0.30)
            self.assertGreaterEqual(upper, 0.65)

        bottle_marker = world.find(".//model[@name='bottle_zone_marker']")
        tennis_marker = world.find(".//model[@name='tennis_zone_marker']")
        bottle_rgb = list(map(float, bottle_marker.findtext(
            ".//visual/material/diffuse"
        ).split()))[:3]
        tennis_rgb = list(map(float, tennis_marker.findtext(
            ".//visual/material/diffuse"
        ).split()))[:3]
        self.assertGreater(bottle_rgb[0], max(bottle_rgb[1:]))
        self.assertGreater(tennis_rgb[1], max(tennis_rgb[0], tennis_rgb[2]))


if __name__ == "__main__":
    unittest.main()
