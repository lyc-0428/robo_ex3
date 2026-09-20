from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


ACTIONS = (
    Path(__file__).resolve().parents[1]
    / "recommended_overlay" / "digital" / "actions"
)
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
        target = standoff_pose(Pose2D(0, 0, 0), 0.47, 0.0, 0.35)
        self.assertAlmostEqual(target.x, 0.12)
        self.assertAlmostEqual(target.y, 0.0)
        self.assertAlmostEqual(target.yaw, 0.0)

    def test_controller_requires_stable_samples(self):
        controller = PoseController(PoseControllerConfig(required_stable_samples=3))
        current = Pose2D(1.0, 2.0, 0.3)
        target = Pose2D(1.0, 2.0, 0.3)
        self.assertFalse(controller.compute(current, target).reached)
        self.assertFalse(controller.compute(current, target).reached)
        self.assertTrue(controller.compute(current, target).reached)

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

    def test_bottle_and_tennis_pivots_are_opposite_and_in_place(self):
        controller = PoseController()
        current = Pose2D(0.0, 0.0, 0.0)
        tennis = controller.compute_yaw(
            current, Pose2D(0.0, 0.0, math.pi / 2.0)
        )
        controller.reset()
        bottle = controller.compute_yaw(
            current, Pose2D(0.0, 0.0, -math.pi / 2.0)
        )
        self.assertEqual(tennis.wheels, tuple(-value for value in bottle.wheels))
        self.assertAlmostEqual(tennis.wheels[0], -tennis.wheels[1])
        self.assertAlmostEqual(bottle.wheels[0], -bottle.wheels[1])

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

    def test_position_only_accepts_five_centimetre_cycle_region(self):
        controller = PoseController(PoseControllerConfig(required_stable_samples=1))
        output = controller.compute_position(
            Pose2D(0.0132, -0.0309, math.radians(43.4)),
            Pose2D(0.0, 0.0, 0.0),
            allow_reverse=True,
            arrival_tolerance=0.050,
        )
        self.assertTrue(output.reached)
        self.assertEqual(output.wheels, (0.0, 0.0, 0.0, 0.0))

    def test_yaw_only_ignores_translation_drift(self):
        controller = PoseController(PoseControllerConfig(required_stable_samples=1))
        output = controller.compute_yaw(
            Pose2D(0.20, -0.10, math.pi / 2.0),
            Pose2D(0.0, 0.0, math.pi / 2.0),
        )
        self.assertTrue(output.reached)
        self.assertEqual(output.phase, "yaw_reached")

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

    def test_closed_loop_converges_without_a_duration_calibration(self):
        controller = PoseController()
        pose = Pose2D(0.0, 0.0, 0.0)
        target = Pose2D(0.16, 0.08, math.radians(35))
        dt = 0.02
        reached = False
        for _ in range(4000):
            output = controller.compute(pose, target)
            if output.reached:
                reached = True
                break
            fl, fr, rl, rr = output.wheels
            linear_wheel = 0.25 * (fl + fr + rl + rr)
            angular_wheel = 0.25 * (-fl + fr - rl + rr)
            linear_mps = 0.05 * linear_wheel
            yaw_rate = 0.80 * angular_wheel
            pose = Pose2D(
                pose.x + linear_mps * math.cos(pose.yaw) * dt,
                pose.y + linear_mps * math.sin(pose.yaw) * dt,
                wrap_angle(pose.yaw + yaw_rate * dt),
            )
        self.assertTrue(reached)
        self.assertLess(math.hypot(pose.x - target.x, pose.y - target.y), 0.011)
        self.assertLess(abs(wrap_angle(pose.yaw - target.yaw)), math.radians(1.1))


if __name__ == "__main__":
    unittest.main()
