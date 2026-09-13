"""Gazebo identity association; class labels always come from YOLO."""
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np


def rotation(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def transform(position, orient):
    t = np.eye(4)
    t[:3, 3] = [position.get(k, 0.0) for k in ('x', 'y', 'z')]
    x, y, z, w = [orient.get(k, 1.0 if k == 'w' else 0.0) for k in ('x', 'y', 'z', 'w')]
    t[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    return t


class SortingIdentity:
    def __init__(self, urdf_path):
        self.joints = list(ET.parse(urdf_path).getroot().findall('joint'))
        self.poses = {}
        self.updated = 0.0
        self.completed = set()
        self.process = subprocess.Popen(
            ['ign', 'topic', '-e', '--json-output', '-t', '/world/pick_place/pose/info'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        decoder = json.JSONDecoder()
        buffer = ''
        for line in self.process.stdout:
            buffer += line
            while buffer.strip():
                buffer = buffer.lstrip()
                try:
                    message, end = decoder.raw_decode(buffer)
                except ValueError:
                    if len(buffer) > 4000000:
                        buffer = ''
                    break
                buffer = buffer[end:]
                poses = {p.get('name'): p for p in message.get('pose', []) if p.get('name')}
                if 'robomaster_ep_core' in poses:
                    self.poses = poses
                    self.updated = time.monotonic()

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)

    def fresh(self):
        return time.monotonic() - self.updated < 2.0

    def position(self, name):
        if not self.fresh() or name not in self.poses:
            return None
        return np.array([self.poses[name].get('position', {}).get(k, 0.0) for k in ('x', 'y', 'z')])

    def yaw(self):
        if not self.fresh():
            return None
        pose = self.poses.get('robomaster_ep_core')
        if pose is None:
            return None
        r = transform({}, pose.get('orientation', {}))
        return math.atan2(r[1, 0], r[0, 0])

    def camera(self, joint_state):
        poses = self.poses
        if not self.fresh() or joint_state is None or 'robomaster_ep_core' not in poses:
            return None
        robot = poses['robomaster_ep_core']
        angles = dict(zip(joint_state.name, joint_state.position))
        children = {j.find('child').get('link') for j in self.joints}
        roots = {j.find('parent').get('link') for j in self.joints} - children
        matrices = {name: transform(robot.get('position', {}), robot.get('orientation', {})) for name in roots}
        pending = list(self.joints)
        while pending:
            progress = False
            for j in pending[:]:
                parent, child = j.find('parent').get('link'), j.find('child').get('link')
                if parent not in matrices:
                    continue
                origin = j.find('origin')
                t = np.eye(4)
                if origin is not None:
                    t[:3, 3] = [float(v) for v in origin.get('xyz', '0 0 0').split()]
                    t[:3, :3] = rotation([float(v) for v in origin.get('rpy', '0 0 0').split()])
                motion = np.eye(4)
                q = angles.get(j.get('name'), 0.0)
                axis_node = j.find('axis')
                axis = np.array([float(v) for v in (axis_node.get('xyz', '1 0 0') if axis_node is not None else '1 0 0').split()])
                if j.get('type') in ('revolute', 'continuous'):
                    axis /= np.linalg.norm(axis)
                    x, y, z = axis
                    skew = np.array([[0,-z,y], [z,0,-x], [-y,x,0]])
                    motion[:3,:3] = np.eye(3) + math.sin(q)*skew + (1-math.cos(q))*(skew @ skew)
                elif j.get('type') == 'prismatic':
                    motion[:3,3] = axis*q
                matrices[child] = matrices[parent] @ t @ motion
                pending.remove(j)
                progress = True
            if not progress:
                return None
        mount = 1.6898531
        forward = float(os.environ.get('TEAM21_CAMERA_FORWARD_OFFSET', '.05'))
        height = float(os.environ.get('TEAM21_CAMERA_HEIGHT_OFFSET', '.20'))
        sensor = np.eye(4)
        sensor[:3,3] = [.002+math.cos(mount)*forward-math.sin(mount)*height, 0,
                        .004+math.sin(mount)*forward+math.cos(mount)*height]
        sensor[:3,:3] = rotation([0, float(os.environ.get('TEAM21_CAMERA_SENSOR_PITCH', '-0.729922011')), 0])
        return matrices['camera_link'] @ sensor

    def associate(self, detections, joint_state, intrinsic, locked=None, counts=None):
        camera = self.camera(joint_state)
        if camera is None:
            return []
        inverse = np.linalg.inv(camera)
        fx, fy, cx, cy = intrinsic
        candidates = []
        for number, detection in enumerate(detections):
            box = detection['bbox']
            u = (box['x1'] + box['x2']) / 2
            v = (box['y1'] + box['y2']) / 2
            width, height = max(12, box['x2']-box['x1']), max(12, box['y2']-box['y1'])
            for name, pose in self.poses.items():
                if not name.startswith('task_object_'):
                    continue
                # Association includes completed entities: never remap their
                # boxes onto a nearby uncompleted entity.
                world = transform(pose.get('position', {}), pose.get('orientation', {}))
                center = np.array([0, 0, .10 if detection['class_name'] == 'bottle' else 0, 1])
                point = inverse @ world @ center
                if point[0] <= .02:
                    continue
                px, py = cx-fx*point[1]/point[0], cy-fy*point[2]/point[0]
                score = ((px-u)/width)**2 + ((py-v)/height)**2
                if abs(px-u) <= .7*width+8 and abs(py-v) <= .7*height+8:
                    candidates.append((score, number, name))
        used_boxes, used_names, result = set(), set(), []
        for score, number, name in sorted(candidates):
            if number in used_boxes or name in used_names:
                continue
            used_boxes.add(number)
            used_names.add(name)
            detection = detections[number]
            if name in self.completed or (locked is not None and name != locked):
                continue
            if counts is not None and counts.get(detection['class_name'], 0) >= 2:
                continue
            result.append(dict(detection, object_id=name))
        return result
