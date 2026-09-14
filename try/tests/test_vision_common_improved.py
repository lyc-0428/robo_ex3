from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET


TRY_ROOT = Path(__file__).resolve().parents[1]
ACTIONS = TRY_ROOT / "recommended_overlay" / "digital" / "actions"
PROJECT = TRY_ROOT.parent
sys.path.insert(0, str(ACTIONS))

from vision_common import (  # noqa: E402
    BOTTLE,
    TENNIS,
    bottle_sdf,
    make_slot_layout,
    tennis_sdf,
    unknown_sdf,
)


class ImprovedSceneTests(unittest.TestCase):
    def test_six_slots_are_balanced_randomized_and_spaced(self):
        angles = [-40, -24, -8, 8, 24, 40]
        first = make_slot_layout(21, 0.47, angles)
        same = make_slot_layout(21, 0.47, angles)
        different = make_slot_layout(22, 0.47, angles)
        self.assertEqual(first, same)
        self.assertNotEqual(
            [item["class_name"] for item in first],
            [item["class_name"] for item in different],
        )
        self.assertEqual(Counter(item["class_name"] for item in first),
                         Counter({BOTTLE: 3, TENNIS: 3}))
        self.assertTrue(all(math.hypot(item["x"], item["y"]) >= 0.47 - 1e-9
                            for item in first))
        separations = [
            math.hypot(a["x"] - b["x"], a["y"] - b["y"])
            for index, a in enumerate(first)
            for b in first[index + 1:]
        ]
        self.assertGreaterEqual(min(separations), 0.12)

    def test_bottle_uses_regular_hexagonal_prism(self):
        model_root = PROJECT / "digital" / "models"
        root = ET.fromstring(bottle_sdf("task_object_0", model_root))
        points = root.findall(".//collision/geometry/polyline/point")
        self.assertEqual(len(points), 6)
        self.assertEqual(root.findtext(".//link/pose"), "0 0 0 0 0 0")
        self.assertEqual(root.findtext(".//inertial/pose"), "0 0 0.100000 0 0 0")
        self.assertEqual(
            root.findtext(".//visual[@name='bottle_visual']/pose"),
            "0 0 -0.003398778 0 0 0",
        )
        self.assertEqual(root.findtext(".//inertial/mass"), "0.200000000")
        self.assertLessEqual(float(root.findtext(".//friction/ode/mu")), 3.0)
        self.assertIsNotNone(root.find(".//visual/geometry/mesh"))
        material_path = (
            TRY_ROOT / "recommended_overlay/digital/models/water_bottle_03/meshes"
            / "bottle_complete.mtl"
        )
        diffuse_colors = [
            tuple(map(float, line.split()[1:4]))
            for line in material_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("Kd ")
        ]
        self.assertTrue(
            diffuse_colors
            and all(blue > green > red for red, green, blue in diffuse_colors)
        )

    def test_tennis_uses_cube_and_unknown_fixture_is_valid(self):
        model_root = PROJECT / "digital" / "models"
        tennis = ET.fromstring(tennis_sdf("task_object_1", model_root))
        self.assertEqual(
            tennis.findtext(".//collision/geometry/box/size"),
            "0.054000 0.054000 0.054000",
        )
        self.assertEqual(
            tennis.findtext(".//visual/geometry/mesh/scale"),
            "0.930000 0.930000 0.930000",
        )
        bottle = ET.fromstring(bottle_sdf("task_object_0", model_root))
        bottle_diffuse = list(map(float, bottle.findtext(
            ".//visual[@name='bottle_visual']/material/diffuse"
        ).split()))[:3]
        self.assertGreater(bottle_diffuse[2], bottle_diffuse[1])
        self.assertGreater(bottle_diffuse[1], bottle_diffuse[0])
        inertia = float(tennis.findtext(".//inertial/inertia/ixx"))
        self.assertGreater(inertia, 0.0)
        unknown = ET.fromstring(unknown_sdf("task_object_5"))
        self.assertIsNotNone(unknown.find(".//collision/geometry/box"))

    def test_too_close_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "separation"):
            make_slot_layout(21, 0.35, [-10, -5, 0, 5, 10, 15])


if __name__ == "__main__":
    unittest.main()
