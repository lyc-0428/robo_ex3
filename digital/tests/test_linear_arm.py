"""Small geometric checks; no ROS, Gazebo, or physical success claims."""

import os
from pathlib import Path
import sys
import unittest

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "actions"))
from linear_arm_kinematics import ARM_JOINTS, ArmReachError, LinearArmKinematics


def source_urdf():
    roots = [PACKAGE, PACKAGE.parents[1] / "work/linear_source/digital"]
    if os.environ.get("DIGITAL_SOURCE"):
        supplied = Path(os.environ["DIGITAL_SOURCE"])
        roots = [supplied, supplied / "digital"] + roots
    for root in roots:
        candidate = root / "src/robomaster_pick_place_sim/urdf/robomaster_ep_static.urdf"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Set DIGITAL_SOURCE to the original digital directory")


class LinearArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names = ARM_JOINTS + ("left_gripper_joint_1", "left_gripper_joint_5",
                                 "right_gripper_joint_1", "right_gripper_joint_5")
        cls.kin = LinearArmKinematics(source_urdf(), cls.names)

    def assertShift(self, before, after, dx=0.0, dz=0.0):
        expected = self.kin.fk(before) + np.array([dx, dz, 0.0])
        actual = self.kin.fk(after)
        self.assertLessEqual(float(np.linalg.norm(actual[:2] - expected[:2])), .0003)
        self.assertLessEqual(abs(float(actual[2] - expected[2])), .003)
        self.assertEqual(list(before[3:]), list(after[3:]))
        self.assertGreaterEqual(after[0], -.274)
        self.assertLessEqual(after[0], 1.384)

    def test_class_grasp_hover_and_test_lift(self):
        # A 0.28005-m row replaces the original 0.01995-m base insertion.
        for category, baseline, dx in (
            ("tennis", [1.38, -.78, -.60, -.32, .71, .32, -.71], 0.0),
            ("bottle", [1.38, -1.38, 0.0, -.25, .43, .25, -.43], -.05),
        ):
            with self.subTest(category=category):
                grasp = self.kin.shift(baseline, dx=dx)
                self.assertShift(baseline, grasp, dx=dx)
                for height in (.025, .08):
                    lifted = self.kin.shift(grasp, dz=height)
                    self.assertShift(grasp, lifted, dz=height)

    def test_carry_preserves_all_fingers(self):
        folded = [0.0, 0.0, 0.0, -.32, .71, .32, -.71]
        carry = self.kin.shift(folded, dx=-.05, dz=-.03)
        self.assertShift(folded, carry, dx=-.05, dz=-.03)
        x, z, _ = self.kin.fk(carry)
        self.assertLess(x, .069)
        self.assertGreater(z, .155)

    def test_unreachable_request_is_an_error(self):
        with self.assertRaises(ArmReachError):
            self.kin.shift([1.38, -.78, -.60, -.32, .71, .32, -.71], dx=.07)
        with self.assertRaises(ArmReachError):
            self.kin.shift([0, 0, 0, -.32, .71, .32, -.71], dx=-.05, dz=.025)

    def test_explicit_names_and_input_limits(self):
        kin = LinearArmKinematics(source_urdf())
        before = [0, 0, 0, -.32, .71, .32, -.71]
        after = kin.shift(before, self.names, dx=-.05, dz=-.03)
        self.assertShift(before, after, dx=-.05, dz=-.03)
        with self.assertRaises(ArmReachError):
            kin.shift([1.50, -.78, -.60, -.32, .71, .32, -.71], self.names)


if __name__ == "__main__":
    unittest.main()
