import sys
from pathlib import Path
import unittest
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'actions'))
from sorting_identity import SortingIdentity


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.identity = SortingIdentity.__new__(SortingIdentity)
        self.identity.completed = set()
        self.identity.camera = lambda state: np.eye(4)
        self.identity.poses = {
            'task_object_0': {'position': {'x': 1, 'y': .3}},
            'task_object_1': {'position': {'x': 1, 'y': -.3}},
        }
        self.boxes = [dict(class_name='tennis', bbox=dict(x1=x-15, x2=x+15, y1=225, y2=255)) for x in (224, 416)]

    def select(self, locked=None, counts=None, class_limits=None):
        return self.identity.associate(
            self.boxes,
            None,
            (320, 320, 320, 240),
            locked,
            counts,
            class_limits,
        )

    def test_completed_instance_stays_excluded_when_both_visible(self):
        self.identity.completed.add('task_object_0')
        self.assertEqual([d['object_id'] for d in self.select()], ['task_object_1'])

    def test_lost_locked_instance_does_not_switch(self):
        self.boxes = self.boxes[1:]
        self.assertEqual(self.select(locked='task_object_0'), [])

    def test_completed_instance_remains_excluded_after_moving_across_image(self):
        self.identity.completed.add('task_object_0')
        self.identity.poses['task_object_0']['position']['y'] = -.3
        self.identity.poses['task_object_1']['position']['y'] = .3
        result = self.select()
        self.assertEqual([d['object_id'] for d in result], ['task_object_1'])
        self.assertEqual(result[0]['bbox']['x1'], 209)

    def test_heading_uses_world_quaternion(self):
        self.identity.updated = time.monotonic()
        self.identity.poses['robomaster_ep_core'] = {
            'orientation': {'z': np.sin(.6), 'w': np.cos(.6)}
        }
        self.assertAlmostEqual(self.identity.yaw(), 1.2)

    def test_class_quota_is_enforced(self):
        self.assertEqual(
            self.select(
                counts={'tennis': 2},
                class_limits={'tennis': 2},
            ),
            [],
        )

    def test_unknown_box_is_not_guessed(self):
        self.boxes = [dict(class_name='tennis', bbox=dict(x1=300,x2=340,y1=225,y2=255))]
        self.assertEqual(self.select(), [])


if __name__ == '__main__':
    unittest.main()
