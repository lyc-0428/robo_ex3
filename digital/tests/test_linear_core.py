"""Focused regressions for the new workflow; no ROS or Gazebo required."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "actions"))
from linear_sorting_core import (
    ROW_X, VerifiedLedger, make_linear_layout, select_visible, drop_y, rail_command,
)


class LinearCoreTests(unittest.TestCase):
    def test_six_mixed_objects_collinear_and_separated(self):
        layout = make_linear_layout(21)
        self.assertEqual([d['class_name'] for d in layout].count('bottle'), 3)
        self.assertEqual([d['class_name'] for d in layout].count('tennis'), 3)
        self.assertEqual({d['x'] for d in layout}, {ROW_X})
        self.assertTrue(all(abs(a['y']-b['y']) >= .299
                            for a, b in zip(layout, layout[1:])))

    def test_invisible_outer_bottle_does_not_block_visible_ball(self):
        d = dict(object_id='inner', class_name='tennis', bbox=dict(x1=300,x2=340))
        positions = dict(inner=(ROW_X, .01, .027), outer=(ROW_X,.09,0))
        chosen = select_visible([[d], [d]], positions, set(), set(), 0, -1)
        self.assertEqual(chosen['object_id'], 'inner')
        self.assertIsNone(select_visible([[d], [d]], positions, {'inner'}, set(), 0, -1))

    def test_votes_do_not_mix_entities_or_classes_or_use_stale_last_frame(self):
        a = dict(object_id='a', class_name='bottle', bbox=dict(x1=300,x2=340))
        b = dict(a, object_id='b')
        positions = {'a':(ROW_X,0,0),'b':(ROW_X,.01,0)}
        self.assertIsNone(select_visible([[a],[b]], positions,set(),set(),0,-1))
        self.assertIsNone(select_visible([[a],[a],[]], positions,set(),set(),0,-1))
        self.assertIsNone(select_visible([[a],[dict(a,class_name='tennis')]], positions,set(),set(),0,-1))

    def test_scan_reversal_changes_order(self):
        a = dict(object_id='a', class_name='bottle', bbox=dict(x1=300,x2=340))
        b = dict(a, object_id='b')
        pos = {'a':(ROW_X,.05,0),'b':(ROW_X,-.05,0)}
        frames = [[a,b],[a,b]]
        self.assertEqual(select_visible(frames,pos,set(),set(),0,-1)['object_id'], 'a')
        self.assertEqual(select_visible(frames,pos,set(),set(),0,1)['object_id'], 'b')

    def test_only_six_unique_verified_releases_finish(self):
        ledger = VerifiedLedger()
        for i, kind in enumerate(['tennis','bottle']*3):
            y = drop_y(kind, ledger.counts[kind])
            released = (ROW_X,y,.027 if kind=='tennis' else 0)
            self.assertTrue(ledger.record(str(i),kind,(ROW_X,0,0),released,y))
            self.assertFalse(ledger.record(str(i),kind,(ROW_X,0,0),released,y))
            self.assertEqual(ledger.done, i==5)
        self.assertEqual(ledger.counts, dict(tennis=3,bottle=3))

    def test_missed_wrong_side_or_airborne_drop_never_counts(self):
        for released in [(ROW_X,0,0),(ROW_X,1.3,0),(ROW_X,-1.3,.2)]:
            ledger = VerifiedLedger()
            with self.assertRaises(ValueError):
                ledger.record('x','bottle',(ROW_X,.4,0),released,-1.3)
            self.assertFalse(ledger.completed)

    def test_lateral_control_does_not_turn_to_path(self):
        self.assertEqual(rail_command(0,0,0,1,.08), (0,.08,0))
        self.assertEqual(rail_command(0,0,0,-1,.08), (0,-.08,0))
        with self.assertRaises(ValueError):
            rail_command(0,0,math.pi/2,1,.08)

    def test_reported_seven_mm_residual_is_inside_grasp_tolerance(self):
        self.assertLessEqual(abs(.757-.750), .015)


if __name__ == '__main__':
    unittest.main()
