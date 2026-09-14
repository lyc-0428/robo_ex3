from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "recommended_overlay" / "digital"
sys.path.insert(0, str(OVERLAY / "actions"))

from task_state_machine import SortingState, SortingStateMachine  # noqa: E402


class OverlayContractTests(unittest.TestCase):
    def test_every_python_file_parses(self):
        for path in list(OVERLAY.rglob("*.py")) + list((ROOT / "alternatives").glob("*.py")):
            with self.subTest(path=path):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_chassis_path_has_no_timed_turn_primitive(self):
        source = (OVERLAY / "actions" / "grasp_bottle_tennis.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_rotate_by_steps", source)
        self.assertNotIn("turn_steps", source)
        self.assertIn("CONFIRM RETURN WITHIN CYCLE CENTER REGION", source)
        self.assertIn("RETURN TO CYCLE YAW", source)
        self.assertIn('control_mode="position"', source)
        self.assertIn('control_mode="yaw"', source)
        self.assertIn("self._identity.pose2d()", source)

    def test_retreat_precedes_exact_turn_and_distant_placement(self):
        source = (OVERLAY / "actions" / "grasp_bottle_tennis.py").read_text(
            encoding="utf-8"
        )
        retreat = source.index("CONFIRM RETREAT WITHIN CYCLE CENTER REGION")
        exact_turn = source.index("TURN EXACTLY 90 DEG")
        distant_place = source.index("DRIVE {placement_distance:.3f} M")
        self.assertLess(retreat, exact_turn)
        self.assertLess(exact_turn, distant_place)
        self.assertIn("allow_reverse=True", source)

    def test_vision_only_runs_during_acquire_and_align(self):
        action = (OVERLAY / "actions" / "grasp_bottle_tennis.py").read_text(
            encoding="utf-8"
        )
        detector = (OVERLAY / "actions" / "yolo_detector.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("/vision_sorting/enable_detection", action)
        self.assertIn("motion phases own control", action)
        self.assertIn("if not self._detection_enabled", detector)

    def test_one_launch_contains_complete_pipeline(self):
        source = (
            OVERLAY
            / "src" / "robomaster_pick_place_sim" / "launch"
            / "vision_sorting_improved.launch.py"
        ).read_text(encoding="utf-8")
        for required in (
            "static_model.launch.py",
            "ros_gz_bridge",
            "yolo_detector.py",
            "grasp_bottle_tennis.py",
            "spawn_sorting_scene.py",
        ):
            self.assertIn(required, source)
        self.assertIn("controller-ready-file", source)
        self.assertIn("scene-ready-file", source)

    def test_standard_detection_interface_and_persistent_log(self):
        source = (OVERLAY / "actions" / "yolo_detector.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Detection2DArray", source)
        self.assertIn('"/detections_2d"', source)
        self.assertIn("detection_log_path", source)

    def test_explicit_nominal_state_path(self):
        machine = SortingStateMachine()
        for state in (
            SortingState.WAIT_SCENE,
            SortingState.ACQUIRE,
            SortingState.ALIGN,
            SortingState.APPROACH,
            SortingState.GRASP,
            SortingState.RETREAT,
            SortingState.PLACE,
            SortingState.VERIFY,
            SortingState.RETURN_HOME,
            SortingState.RECORD,
            SortingState.DONE,
        ):
            machine.transition(state)
        self.assertEqual(machine.state, SortingState.DONE)

    def test_world_and_parameter_files_are_valid(self):
        world = OVERLAY / "src" / "robomaster_pick_place_sim" / "worlds" / "pick_place.sdf"
        ET.parse(world)
        package = ET.parse(
            OVERLAY / "src" / "robomaster_pick_place_sim" / "package.xml"
        )
        dependencies = [node.text for node in package.findall(".//exec_depend")]
        self.assertIn("vision_msgs", dependencies)
        params = (
            OVERLAY / "src" / "robomaster_pick_place_sim" / "config"
            / "vision_sorting_improved.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("total_slot_count: 6", params)
        self.assertIn("minimum_success_count: 5", params)
        self.assertIn("position_tolerance_m: 0.010", params)
        self.assertIn("placement_yaw_degrees: 90.0", params)
        self.assertIn("placement_distance_m: 0.850", params)
        self.assertIn("placement_distance_decrement_m: 0.120", params)
        self.assertIn("placement_minimum_distance_m: 0.250", params)
        self.assertIn("cycle_return_tolerance_m: 0.015", params)
        self.assertIn("minimum_grasp_approach_m: 0.040", params)

    def test_missed_grasp_state_path_returns_before_fresh_acquire(self):
        machine = SortingStateMachine()
        for state in (
            SortingState.WAIT_SCENE,
            SortingState.ACQUIRE,
            SortingState.ALIGN,
            SortingState.APPROACH,
            SortingState.GRASP,
            SortingState.VERIFY,
            SortingState.RETURN_HOME,
            SortingState.RECORD,
            SortingState.ACQUIRE,
        ):
            machine.transition(state)
        self.assertEqual(machine.state, SortingState.ACQUIRE)

        source = (OVERLAY / "actions" / "grasp_bottle_tennis.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("ROUND RETRY", source)
        self.assertIn("retry round with fresh visual acquisition", source)
        self.assertIn("success_count_unchanged=true", source)

    def test_exception_scenarios_exist(self):
        spawn = (OVERLAY / "actions" / "spawn_sorting_scene.py").read_text(
            encoding="utf-8"
        )
        action = (OVERLAY / "actions" / "grasp_bottle_tennis.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '(\"nominal\", \"unknown\", \"empty\", \"tennis_only\")',
            spawn,
        )
        self.assertIn("EXCEPTION UNRECOGNIZED", action)
        self.assertIn("EXCEPTION EMPTY GRID", action)

    def test_restored_bottle_detector_contract(self):
        detector = (OVERLAY / "actions/yolo_detector.py").read_text(
            encoding="utf-8"
        )
        launch = (
            OVERLAY / "src/robomaster_pick_place_sim/launch"
            / "vision_sorting_improved.launch.py"
        ).read_text(encoding="utf-8")
        runner = (OVERLAY / "run_vision_sorting_improved.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('DeclareLaunchArgument("bottle_confidence", default_value="0.40")', launch)
        self.assertIn("adjusted_bottle_confidence", detector)
        self.assertIn('"color_bonus"', detector)
        self.assertIn('elif [[ "${SCENARIO}" == "nominal" ]]', runner)


if __name__ == "__main__":
    unittest.main()
