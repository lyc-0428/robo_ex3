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
        self.assertTrue(all(math.isclose(item["z"], 0.029) for item in layout))
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
        self.assertIn("PLACEMENT MOTION VERIFIED", action)


if __name__ == "__main__":
    unittest.main()
