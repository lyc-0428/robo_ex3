from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


DIGITAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIGITAL / "actions"))

from placement_zone import (  # noqa: E402
    in_classification_half,
    lateral_offset,
    placement_distance_for_attempt,
)


class PlacementZoneTests(unittest.TestCase):
    def test_tennis_accepts_any_point_safely_inside_left_half(self):
        self.assertTrue(in_classification_half(
            0.0, 0.0, 0.0, 0.08, 0.31, left_half=True, margin=0.20
        ))
        self.assertFalse(in_classification_half(
            0.0, 0.0, 0.0, -0.05, 0.19, left_half=True, margin=0.20
        ))

    def test_bottle_accepts_any_point_safely_inside_right_half(self):
        self.assertTrue(in_classification_half(
            0.0, 0.0, 0.0, -0.12, -0.27, left_half=False, margin=0.20
        ))
        self.assertFalse(in_classification_half(
            0.0, 0.0, 0.0, 0.02, 0.30, left_half=False, margin=0.20
        ))

    def test_half_planes_rotate_with_cycle_heading(self):
        offset = lateral_offset(
            1.0, 2.0, math.pi / 2.0, 0.70, 2.02
        )
        self.assertAlmostEqual(offset, 0.30, places=6)

    def test_margin_must_be_positive(self):
        with self.assertRaises(ValueError):
            in_classification_half(
                0.0, 0.0, 0.0, 0.0, 0.1, left_half=True, margin=0.0
            )

    def test_six_successful_placements_get_distinct_shorter_distances(self):
        distances = [
            placement_distance_for_attempt(0.65, 0.07, 0.30, attempt)
            for attempt in range(1, 7)
        ]
        self.assertEqual(
            [round(distance, 2) for distance in distances],
            [0.65, 0.58, 0.51, 0.44, 0.37, 0.30],
        )

    def test_placement_distance_is_clamped_after_six(self):
        self.assertAlmostEqual(
            placement_distance_for_attempt(0.65, 0.07, 0.30, 20),
            0.30,
        )

    def test_placement_distance_rejects_invalid_parameters(self):
        invalid = (
            (0.0, 0.04, 0.30, 1),
            (0.50, -0.01, 0.30, 1),
            (0.30, 0.04, 0.50, 1),
            (0.50, 0.04, 0.30, 0),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                placement_distance_for_attempt(*values)


if __name__ == "__main__":
    unittest.main()
