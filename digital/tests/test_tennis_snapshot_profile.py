from __future__ import annotations

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


DIGITAL = Path(__file__).resolve().parents[1]
PROFILE = DIGITAL / "actions" / "tennis_debug"


class TennisSnapshotProfileTests(unittest.TestCase):
    def test_profile_has_private_copies_of_runtime_modules(self):
        required = {
            "grasp_bottle_tennis.py",
            "grasp_cube.py",
            "gripper_staging.py",
            "placement_zone.py",
            "pose_feedback.py",
            "sorting_identity.py",
            "spawn_sorting_scene.py",
            "task_state_machine.py",
            "vision_common.py",
            "yolo_detector.py",
        }
        self.assertTrue(required.issubset({path.name for path in PROFILE.iterdir()}))

    def test_requested_58_test_snapshot_tuning_is_frozen(self):
        action = (PROFILE / "grasp_bottle_tennis.py").read_text(encoding="utf-8")
        config = (
            DIGITAL / "src/robomaster_pick_place_sim/config"
            / "vision_sorting_tennis_debug.yaml"
        ).read_text(encoding="utf-8")
        self.assertNotIn("COARSE WORLD-BEARING ALIGN", action)
        self.assertNotIn("_sorting_home_pose", action)
        self.assertNotIn("UNSAFE SHORT APPROACH REJECTED", action)
        detector = (PROFILE / "yolo_detector.py").read_text(encoding="utf-8")
        self.assertNotIn("adjusted_bottle_confidence", detector)
        self.assertIn("cycle_return_tolerance_m: 0.050", config)
        self.assertIn("placement_distance_m: 0.650", config)
        self.assertIn("placement_distance_decrement_m: 0.070", config)
        self.assertIn("placement_minimum_distance_m: 0.300", config)

    def test_tennis_geometry_and_markers_match_snapshot(self):
        common = (PROFILE / "vision_common.py").read_text(encoding="utf-8")
        self.assertIn("TENNIS_COLLISION_SIDE = 0.054", common)
        self.assertIn("TENNIS_VISUAL_SCALE = 0.93", common)
        world = ET.parse(
            DIGITAL / "src/robomaster_pick_place_sim/worlds"
            / "pick_place_tennis_debug.sdf"
        )
        for name in ("bottle_zone_marker", "tennis_zone_marker"):
            self.assertEqual(
                world.findtext(
                    f".//model[@name='{name}']/link/visual/geometry/box/size"
                ),
                "1.20 0.80 0.002",
            )

    def test_runner_and_launch_use_only_dedicated_profile(self):
        runner = (DIGITAL / "run_tennis_placement_debug.sh").read_text(
            encoding="utf-8"
        )
        launch = (
            DIGITAL / "src/robomaster_pick_place_sim/launch"
            / "vision_sorting_tennis_debug.launch.py"
        ).read_text(encoding="utf-8")
        static_launch = (
            DIGITAL / "src/robomaster_pick_place_sim/launch"
            / "static_model_tennis_debug.launch.py"
        ).read_text(encoding="utf-8")
        self.assertIn("vision_sorting_tennis_debug.launch.py", runner)
        self.assertNotIn("vision_sorting_improved.launch.py", runner)
        self.assertIn("models/tennis_debug", runner)
        self.assertIn('ROBO_EX3_TENNIS_ROS_DOMAIN_ID:-122', runner)
        self.assertIn('[digital_root, "actions", "tennis_debug"]', launch)
        self.assertIn("vision_sorting_tennis_debug.yaml", launch)
        self.assertIn("static_model_tennis_debug.launch.py", launch)
        self.assertIn("pick_place_tennis_debug.sdf", static_launch)


if __name__ == "__main__":
    unittest.main()
