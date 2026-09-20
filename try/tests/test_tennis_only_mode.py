from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


DIGITAL = (
    Path(__file__).resolve().parents[1]
    / "recommended_overlay/digital"
)
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

    def test_locked_approach_is_straight_and_retry_reacquires(self):
        action = (DIGITAL / "actions/grasp_bottle_tennis.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("locked_heading = current.yaw", action)
        self.assertIn("STRAIGHT GRASP CORRIDOR LOCKED", action)
        self.assertNotIn("FACE LOCKED OBJECT", action)
        self.assertIn("RECOVER MISSED GRASP", action)
        self.assertIn("returned_to_origin=true | reacquire=true", action)

    def test_six_placement_distances_are_configured(self):
        config = (
            DIGITAL
            / "src/robomaster_pick_place_sim/config/vision_sorting_improved.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("placement_distance_m: 0.850", config)
        self.assertIn("placement_distance_decrement_m: 0.120", config)
        self.assertIn("placement_minimum_distance_m: 0.250", config)

    def test_small_home_residual_does_not_start_search_rotation(self):
        action = (DIGITAL / "actions/grasp_bottle_tennis.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("CONFIRM RETREAT WITHIN CYCLE CENTER REGION", action)
        self.assertIn("DIRECT CLASSIFICATION TURN", action)
        self.assertNotIn("RESTORE CYCLE HEADING WITH OBJECT", action)

    def test_markers_are_large_and_use_red_green_class_colors(self):
        import xml.etree.ElementTree as ET

        world = ET.parse(
            DIGITAL / "src/robomaster_pick_place_sim/worlds/pick_place.sdf"
        )
        bottle = world.find(".//model[@name='bottle_zone_marker']")
        tennis = world.find(".//model[@name='tennis_zone_marker']")
        for marker in (bottle, tennis):
            self.assertEqual(
                marker.findtext(".//visual/geometry/box/size"),
                "1.40 0.90 0.002",
            )
        bottle_rgb = list(map(float, bottle.findtext(
            ".//visual/material/diffuse"
        ).split()))[:3]
        tennis_rgb = list(map(float, tennis.findtext(
            ".//visual/material/diffuse"
        ).split()))[:3]
        self.assertGreater(bottle_rgb[0], max(bottle_rgb[1:]))
        self.assertGreater(tennis_rgb[1], max(tennis_rgb[0], tennis_rgb[2]))

    def test_fixed_home_retry_and_short_approach_guard_are_present(self):
        action = (DIGITAL / "actions/grasp_bottle_tennis.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("self._cycle_pose = self._sorting_home_pose", action)
        self.assertIn("FINAL FIXED HOME POSE VERIFICATION", action)
        self.assertIn("VISION ALIGN RETRY", action)
        self.assertIn("UNSAFE SHORT APPROACH REJECTED", action)
        self.assertIn("COARSE WORLD-BEARING ALIGN", action)

    def test_nominal_bottle_and_tennis_turns_are_opposite_pivots(self):
        action = (DIGITAL / "actions/grasp_bottle_tennis.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "-self.placement_yaw if self._current_class == BOTTLE",
            action,
        )
        self.assertIn("CLASS-SPECIFIC IN-PLACE TURN", action)
        self.assertIn("yaw controller generated non-pivot wheels", action)


if __name__ == "__main__":
    unittest.main()
