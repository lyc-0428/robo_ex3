"""Coordinate feedback alternatives for simulation and the real chassis."""

from __future__ import annotations

from dataclasses import dataclass
import math


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


class WheelImuOdometry:
    """Fuse wheel travel with an absolute IMU yaw measurement.

    This is the real-car fallback when no external marker is available.  It
    removes time-based turning but still accumulates translation drift.
    """

    def __init__(self, wheel_radius: float, track_width: float,
                 imu_weight: float = 0.85):
        if min(wheel_radius, track_width) <= 0.0:
            raise ValueError("wheel geometry must be positive")
        if not 0.0 <= imu_weight <= 1.0:
            raise ValueError("imu_weight must be in [0, 1]")
        self.wheel_radius = wheel_radius
        self.track_width = track_width
        self.imu_weight = imu_weight
        self.pose = Pose2D(0.0, 0.0, 0.0)

    def update(self, wheel_delta_radians, imu_yaw: float) -> Pose2D:
        if len(wheel_delta_radians) != 4:
            raise ValueError("four wheel deltas are required")
        fl, fr, rl, rr = [float(value) for value in wheel_delta_radians]
        left = 0.5 * (fl + rl) * self.wheel_radius
        right = 0.5 * (fr + rr) * self.wheel_radius
        distance = 0.5 * (left + right)
        encoder_yaw = wrap(self.pose.yaw + (right - left) / self.track_width)
        yaw_error = wrap(float(imu_yaw) - encoder_yaw)
        fused_yaw = wrap(encoder_yaw + self.imu_weight * yaw_error)
        midpoint = wrap(0.5 * (self.pose.yaw + fused_yaw))
        self.pose = Pose2D(
            self.pose.x + distance * math.cos(midpoint),
            self.pose.y + distance * math.sin(midpoint),
            fused_yaw,
        )
        return self.pose


class AbsolutePoseCorrector:
    """Blend odometry with AprilTag, ArUco, Vicon, or Gazebo world pose."""

    def __init__(self, correction_weight=1.0):
        if not 0.0 < correction_weight <= 1.0:
            raise ValueError("correction_weight must be in (0, 1]")
        self.weight = correction_weight

    def correct(self, estimate: Pose2D, absolute: Pose2D) -> Pose2D:
        yaw_error = wrap(absolute.yaw - estimate.yaw)
        return Pose2D(
            estimate.x + self.weight * (absolute.x - estimate.x),
            estimate.y + self.weight * (absolute.y - estimate.y),
            wrap(estimate.yaw + self.weight * yaw_error),
        )


def return_error(current: Pose2D, home: Pose2D) -> tuple[float, float]:
    return (
        math.hypot(current.x - home.x, current.y - home.y),
        abs(wrap(current.yaw - home.yaw)),
    )


def is_home(current: Pose2D, home: Pose2D,
            position_tolerance=0.01, yaw_tolerance=math.radians(1.0)) -> bool:
    position_error, yaw_error = return_error(current, home)
    return position_error <= position_tolerance and yaw_error <= yaw_tolerance

