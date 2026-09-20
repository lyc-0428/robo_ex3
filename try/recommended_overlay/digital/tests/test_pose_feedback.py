from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


ACTIONS = Path(__file__).resolve().parents[1] / "actions"
sys.path.insert(0, str(ACTIONS))

from pose_feedback import (  # noqa: E402
    Pose2D,
    PoseController,
    PoseControllerConfig,
    standoff_pose,
    wrap_angle,
)


class PoseFeedbackTests(unittest.TestCase):
    def test_angle_wrap_chooses_short_path(self):
        self.assertAlmostEqual(wrap_angle(math.radians(181)), math.radians(-179))

    def test_standoff_is_on_target_ray(self):
        target = standoff_pose(Pose2D(0.0, 0.0, 0.0), 1.0, 0.0, 0.2)
        self.assertAlmostEqual(target.x, 0.8)
        self.assertAlmostEqual(target.y, 0.0)
        self.assertAlmostEqual(target.yaw, 0.0)

    def test_controller_requires_stable_samples(self):
        controller = PoseController(PoseControllerConfig(required_stable_samples=4))
        current = Pose2D(0.0, 0.0, 0.0)
        reached = [controller.compute(current, current).reached for _ in range(4)]
        self.assertEqual(reached, [False, False, False, True])

    def test_reverse_return_does_not_turn_around_at_the_object(self):
        controller = PoseController()
        output = controller.compute(
            Pose2D(0.20, 0.0, 0.0),
            Pose2D(0.0, 0.0, 0.0),
            allow_reverse=True,
        )
        self.assertEqual(output.phase, "reverse_to_position")
        self.assertTrue(all(wheel < 0.0 for wheel in output.wheels))

    def test_same_position_with_ninety_degree_target_must_turn(self):
        controller = PoseController()
        output = controller.compute(
            Pose2D(0.0, 0.0, 0.0),
            Pose2D(0.0, 0.0, math.pi / 2.0),
        )
        self.assertEqual(output.phase, "turn_to_final_yaw")
        self.assertFalse(output.reached)
        self.assertLess(output.wheels[0], 0.0)
        self.assertGreater(output.wheels[1], 0.0)

    def test_final_yaw_hysteresis_ignores_one_centimetre_pose_noise(self):
        controller = PoseController()
        target = Pose2D(0.0, 0.0, math.pi / 2.0)
        controller.compute(Pose2D(0.009, 0.0, 0.0), target)
        output = controller.compute(Pose2D(0.0105, 0.0, 0.1), target)
        self.assertEqual(output.phase, "turn_to_final_yaw")

    def test_final_position_is_still_strict_after_yaw_converges(self):
        controller = PoseController()
        target = Pose2D(0.0, 0.0, math.pi / 2.0)
        controller.compute(Pose2D(0.009, 0.0, 0.0), target)
        output = controller.compute(Pose2D(0.0105, 0.0, math.pi / 2.0), target)
        self.assertTrue(output.phase.startswith("reacquire_"))
        self.assertFalse(output.reached)

    def test_position_only_does_not_require_target_yaw(self):
        controller = PoseController(PoseControllerConfig(required_stable_samples=1))
        output = controller.compute_position(
            Pose2D(0.0, 0.0, 0.0),
            Pose2D(0.0, 0.0, math.pi),
        )
        self.assertTrue(output.reached)
        self.assertEqual(output.phase, "position_reached")

    def test_position_only_can_accept_a_broad_cycle_return_region(self):
        controller = PoseController(PoseControllerConfig(required_stable_samples=1))
        output = controller.compute_position(
            Pose2D(0.0132, -0.0309, math.radians(43.4)),
            Pose2D(0.0, 0.0, 0.0),
            allow_reverse=True,
            arrival_tolerance=0.050,
        )
        self.assertTrue(output.reached)
        self.assertEqual(output.phase, "position_reached")
        self.assertEqual(output.wheels, (0.0, 0.0, 0.0, 0.0))

    def test_yaw_only_ignores_translation_drift(self):
        controller = PoseController(PoseControllerConfig(required_stable_samples=1))
        output = controller.compute_yaw(
            Pose2D(0.20, -0.10, math.pi / 2.0),
            Pose2D(0.0, 0.0, math.pi / 2.0),
        )
        self.assertTrue(output.reached)
        self.assertEqual(output.phase, "yaw_reached")

    def test_straight_mode_keeps_fixed_heading_near_endpoint(self):
        controller = PoseController(PoseControllerConfig(required_stable_samples=1))
        output = controller.compute_straight(
            Pose2D(0.01, 0.439, math.radians(35.0)),
            Pose2D(0.0, 0.450, math.pi / 2.0),
            travel_yaw=math.pi / 2.0,
            arrival_tolerance=0.020,
        )
        self.assertTrue(output.reached)
        self.assertEqual(output.phase, "straight_reached")
        self.assertEqual(output.wheels, (0.0, 0.0, 0.0, 0.0))

    def test_straight_reverse_does_not_turn_toward_origin(self):
        controller = PoseController(PoseControllerConfig(required_stable_samples=1))
        output = controller.compute_straight(
            Pose2D(0.0, 0.450, math.pi / 2.0),
            Pose2D(0.0, 0.0, math.pi / 2.0),
            travel_yaw=math.pi / 2.0,
            arrival_tolerance=0.020,
        )
        self.assertEqual(output.phase, "straight_reverse")
        self.assertTrue(all(wheel < 0.0 for wheel in output.wheels))

    def test_straight_mode_ignores_subdegree_heading_noise(self):
        controller = PoseController()
        output = controller.compute_straight(
            Pose2D(0.0, 0.0, math.radians(0.6)),
            Pose2D(0.4, 0.0, 0.0),
            travel_yaw=0.0,
            arrival_tolerance=0.012,
        )
        self.assertEqual(output.phase, "straight_forward")
        self.assertEqual(len(set(output.wheels)), 1)

    def test_closed_loop_converges_without_duration_calibration(self):
        controller = PoseController(PoseControllerConfig(required_stable_samples=4))
        target = Pose2D(0.6, -0.2, math.radians(35.0))
        current = Pose2D(0.0, 0.0, 0.0)
        reached = False
        dt = 0.02
        for _ in range(4000):
            output = controller.compute(current, target)
            if output.reached:
                reached = True
                break
            left = 0.5 * (output.wheels[0] + output.wheels[2])
            right = 0.5 * (output.wheels[1] + output.wheels[3])
            linear = 0.11 * (left + right) * 0.5
            angular = 0.9 * (right - left) * 0.5
            current = Pose2D(
                current.x + linear * math.cos(current.yaw) * dt,
                current.y + linear * math.sin(current.yaw) * dt,
                wrap_angle(current.yaw + angular * dt),
            )
        self.assertTrue(reached)
        self.assertLessEqual(
            math.hypot(target.x - current.x, target.y - current.y),
            0.010,
        )
        self.assertLessEqual(abs(wrap_angle(target.yaw - current.yaw)), math.radians(1.0))


if __name__ == "__main__":
    unittest.main()
