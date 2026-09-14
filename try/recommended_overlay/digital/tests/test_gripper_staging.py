from __future__ import annotations

from pathlib import Path
import sys
import unittest


DIGITAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIGITAL / "actions"))

from gripper_staging import (  # noqa: E402
    LEFT_ROOT,
    LEFT_TIP,
    RIGHT_ROOT,
    RIGHT_TIP,
    is_tip_first_closing,
    tennis_closed_pose,
    tennis_open_pose,
    tip_first_pose,
)


class GripperStagingTests(unittest.TestCase):
    def setUp(self):
        self.joints = (LEFT_ROOT, RIGHT_ROOT, LEFT_TIP, RIGHT_TIP, "arm")
        self.index = {name: i for i, name in enumerate(self.joints)}
        self.initial = [0.69, -0.70, 1.09, -1.07, 0.25]
        self.opened = [1.39, -1.40, 1.09, -1.07, 0.25]

    def test_tip_guard_is_mirrored_inward(self):
        ordinary_closed = list(self.initial)
        closed = tennis_closed_pose(
            ordinary_closed, self.initial, self.index, 0.28
        )
        self.assertAlmostEqual(closed[self.index[LEFT_TIP]], 0.81)
        self.assertAlmostEqual(closed[self.index[RIGHT_TIP]], -0.79)

    def test_first_phase_moves_tips_but_keeps_roots_open(self):
        closed = tennis_closed_pose(
            self.initial, self.initial, self.index, 0.28
        )
        guarded = tip_first_pose(self.opened, closed, self.index)
        self.assertEqual(guarded[self.index[LEFT_ROOT]], 1.39)
        self.assertEqual(guarded[self.index[RIGHT_ROOT]], -1.40)
        self.assertEqual(guarded[self.index[LEFT_TIP]], closed[self.index[LEFT_TIP]])
        self.assertEqual(guarded[self.index[RIGHT_TIP]], closed[self.index[RIGHT_TIP]])
        self.assertTrue(is_tip_first_closing(self.opened, closed, self.index))

    def test_release_is_not_mistaken_for_staged_close(self):
        closed = tennis_closed_pose(
            self.initial, self.initial, self.index, 0.28
        )
        released = tennis_open_pose(self.opened, self.initial, self.index)
        self.assertEqual(released[self.index[LEFT_TIP]], 1.09)
        self.assertEqual(released[self.index[RIGHT_TIP]], -1.07)
        self.assertFalse(is_tip_first_closing(closed, released, self.index))

    def test_negative_tip_offset_is_rejected(self):
        with self.assertRaises(ValueError):
            tennis_closed_pose(self.initial, self.initial, self.index, -0.01)


if __name__ == "__main__":
    unittest.main()
