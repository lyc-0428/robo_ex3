from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


ALTERNATIVES = Path(__file__).resolve().parents[1] / "alternatives"
sys.path.insert(0, str(ALTERNATIVES))

from collision_strategies import (  # noqa: E402
    box_inertia,
    collision_geometry,
    contact_surface,
)
from heading_strategies import YawPid, visual_servo_turn  # noqa: E402
from layout_strategies import (  # noqa: E402
    minimum_separation,
    randomized_balanced_classes,
    staggered_rows,
    two_arcs,
    wide_arc,
)
from localization_strategies import (  # noqa: E402
    AbsolutePoseCorrector,
    Pose2D,
    WheelImuOdometry,
    is_home,
)


class AlternativeStrategyTests(unittest.TestCase):
    def test_collision_options_and_physics(self):
        self.assertIn("<box>", collision_geometry("box", 0.06, 0.10))
        self.assertEqual(collision_geometry("hex", 0.06, 0.10).count("<point>"), 6)
        self.assertEqual(collision_geometry("octagon", 0.06, 0.10).count("<point>"), 8)
        inertia = box_inertia(0.2, 0.06, 0.06, 0.10)
        self.assertGreater(min(inertia.ixx, inertia.iyy, inertia.izz), 0.0)
        self.assertIn("<mu>1.3000</mu>", contact_surface())

    def test_all_layout_options_have_six_well_separated_slots(self):
        for factory in (wide_arc, staggered_rows, two_arcs):
            slots = factory()
            self.assertEqual(len(slots), 6)
            self.assertGreaterEqual(minimum_separation(slots), 0.12)
        labels = randomized_balanced_classes(21)
        self.assertEqual(labels.count("bottle"), 3)
        self.assertEqual(labels.count("tennis"), 3)

    def test_encoder_imu_and_absolute_correction(self):
        odom = WheelImuOdometry(0.05, 0.30)
        pose = odom.update([1.0, 1.0, 1.0, 1.0], 0.0)
        self.assertGreater(pose.x, 0.0)
        corrected = AbsolutePoseCorrector(1.0).correct(
            pose, Pose2D(0.0, 0.0, 0.0)
        )
        self.assertTrue(is_home(corrected, Pose2D(0.0, 0.0, 0.0)))

    def test_heading_feedback_is_directional(self):
        pid = YawPid(kd=0.0)
        self.assertGreater(pid.update(math.radians(20), 0.0, 0.02), 0.0)
        self.assertLess(visual_servo_turn(100.0), 0.0)
        self.assertEqual(visual_servo_turn(5.0), 0.0)


if __name__ == "__main__":
    unittest.main()

