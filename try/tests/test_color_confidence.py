from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


OVERLAY = Path(__file__).resolve().parents[1] / "recommended_overlay/digital"
sys.path.insert(0, str(OVERLAY / "actions"))

from color_confidence import adjusted_bottle_confidence  # noqa: E402


class ColorConfidenceTests(unittest.TestCase):
    def test_light_blue_adds_exactly_point_five_and_caps_at_one(self):
        image = np.full((12, 12, 3), (235, 184, 122), dtype=np.uint8)
        bbox = {"x1": 0, "y1": 0, "x2": 12, "y2": 12}
        self.assertEqual(
            adjusted_bottle_confidence(image, bbox, 0.20),
            (0.70, 1.0, 0.5),
        )
        self.assertEqual(
            adjusted_bottle_confidence(image, bbox, 0.80),
            (1.0, 1.0, 0.5),
        )

    def test_non_blue_roi_does_not_inflate_confidence(self):
        image = np.full((12, 12, 3), (170, 170, 170), dtype=np.uint8)
        bbox = {"x1": 0, "y1": 0, "x2": 12, "y2": 12}
        self.assertEqual(
            adjusted_bottle_confidence(image, bbox, 0.39),
            (0.39, 0.0, 0.0),
        )
