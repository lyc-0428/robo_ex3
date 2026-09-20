from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ACTIONS = Path(__file__).resolve().parents[1] / "actions"
sys.path.insert(0, str(ACTIONS))

from color_confidence import (  # noqa: E402
    adjusted_bottle_confidence,
    light_blue_fraction_bgr,
)


class ColorConfidenceTests(unittest.TestCase):
    def test_pale_blue_roi_gets_exact_half_point_bonus(self):
        frame = np.full((20, 20, 3), (235, 184, 122), dtype=np.uint8)
        bbox = {"x1": 2, "y1": 2, "x2": 18, "y2": 18}
        adjusted, fraction, bonus = adjusted_bottle_confidence(
            frame, bbox, 0.22
        )
        self.assertEqual(fraction, 1.0)
        self.assertEqual(bonus, 0.5)
        self.assertAlmostEqual(adjusted, 0.72)

    def test_grey_or_dark_blue_does_not_receive_bonus(self):
        bbox = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        for color in ((180, 180, 180), (100, 40, 10)):
            with self.subTest(color=color):
                frame = np.full((10, 10, 3), color, dtype=np.uint8)
                adjusted, fraction, bonus = adjusted_bottle_confidence(
                    frame, bbox, 0.42
                )
                self.assertEqual(fraction, 0.0)
                self.assertEqual(bonus, 0.0)
                self.assertEqual(adjusted, 0.42)

    def test_bbox_is_clipped_and_empty_bbox_is_safe(self):
        frame = np.full((8, 8, 3), (235, 184, 122), dtype=np.uint8)
        self.assertEqual(
            light_blue_fraction_bgr(
                frame, {"x1": -5, "y1": -5, "x2": 4, "y2": 4}
            ),
            1.0,
        )
        self.assertEqual(
            light_blue_fraction_bgr(
                frame, {"x1": 5, "y1": 5, "x2": 5, "y2": 5}
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
