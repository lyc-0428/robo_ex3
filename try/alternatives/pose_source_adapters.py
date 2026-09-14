"""Runtime adapter sketches for the coordinate sources discussed in the report."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


class GazeboWorldPoseSource:
    """Recommended simulation source; wraps the overlay SortingIdentity."""

    def __init__(self, sorting_identity):
        self.identity = sorting_identity

    def read(self):
        return self.identity.pose2d()


class RosOdometryPoseSource:
    """ROS ``nav_msgs/Odometry`` adapter without a module-level ROS import."""

    def __init__(self, node, topic="/odom"):
        from nav_msgs.msg import Odometry

        self._lock = threading.Lock()
        self._pose = None
        node.create_subscription(Odometry, topic, self._callback, 20)

    def _callback(self, message):
        p = message.pose.pose.position
        q = message.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        with self._lock:
            self._pose = Pose2D(float(p.x), float(p.y), yaw)

    def read(self):
        with self._lock:
            return self._pose


class RoboMasterSdkPoseSource:
    """Real-car callback buffer for ``sub_position`` plus ``sub_attitude``."""

    def __init__(self, chassis, frequency=20):
        self._lock = threading.Lock()
        self._xy = None
        self._yaw = None
        chassis.sub_position(freq=frequency, callback=self._position)
        chassis.sub_attitude(freq=frequency, callback=self._attitude)

    def _position(self, values):
        with self._lock:
            self._xy = (float(values[0]), float(values[1]))

    def _attitude(self, values):
        with self._lock:
            self._yaw = math.radians(float(values[0]))

    def read(self):
        with self._lock:
            if self._xy is None or self._yaw is None:
                return None
            return Pose2D(self._xy[0], self._xy[1], self._yaw)

