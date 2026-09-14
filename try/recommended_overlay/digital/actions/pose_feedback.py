#!/usr/bin/env python3
"""Pure feedback helpers for repeatable RoboMaster chassis motion.

The module intentionally has no ROS dependency so the controller can be unit
tested on a development computer.  A runtime node supplies measured world or
odometry poses and publishes the returned wheel velocities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class PoseControllerConfig:
    position_tolerance: float = 0.010
    position_reacquire_tolerance: float = 0.025
    yaw_tolerance: float = math.radians(1.0)
    required_stable_samples: int = 4
    linear_kp: float = 8.0
    angular_kp: float = 4.0
    min_linear_wheel_speed: float = 0.35
    max_linear_wheel_speed: float = 2.0
    min_angular_wheel_speed: float = 0.30
    max_angular_wheel_speed: float = 1.8
    rotate_in_place_threshold: float = math.radians(28.0)
    yaw_wheel_gain: float = 1.0
    max_wheel_speed: float = 3.0

    def validate(self) -> None:
        positive = (
            self.position_tolerance,
            self.yaw_tolerance,
            self.linear_kp,
            self.angular_kp,
            self.max_linear_wheel_speed,
            self.max_angular_wheel_speed,
            self.yaw_wheel_gain,
            self.max_wheel_speed,
        )
        if min(positive) <= 0.0:
            raise ValueError("pose controller gains and limits must be positive")
        if self.required_stable_samples < 1:
            raise ValueError("required_stable_samples must be at least one")
        if self.min_linear_wheel_speed > self.max_linear_wheel_speed:
            raise ValueError("minimum linear speed exceeds maximum")
        if self.min_angular_wheel_speed > self.max_angular_wheel_speed:
            raise ValueError("minimum angular speed exceeds maximum")
        if self.position_reacquire_tolerance <= self.position_tolerance:
            raise ValueError(
                "position reacquire tolerance must exceed position tolerance"
            )


@dataclass(frozen=True)
class ControlOutput:
    wheels: tuple[float, float, float, float]
    position_error: float
    yaw_error: float
    reached: bool
    phase: str


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _signed_floor(value: float, minimum: float) -> float:
    if abs(value) < 1.0e-12:
        return 0.0
    return math.copysign(max(abs(value), minimum), value)


def _clamp(value: float, magnitude: float) -> float:
    return max(-magnitude, min(magnitude, value))


def standoff_pose(
    robot: Pose2D,
    object_x: float,
    object_y: float,
    standoff_distance: float,
) -> Pose2D:
    """Return a base pose on the robot-to-object ray, facing the object."""
    standoff_distance = float(standoff_distance)
    if standoff_distance <= 0.0:
        raise ValueError("standoff distance must be positive")
    dx = float(object_x) - robot.x
    dy = float(object_y) - robot.y
    distance = math.hypot(dx, dy)
    if distance <= standoff_distance:
        return Pose2D(robot.x, robot.y, math.atan2(dy, dx))
    ux = dx / distance
    uy = dy / distance
    return Pose2D(
        float(object_x) - ux * standoff_distance,
        float(object_y) - uy * standoff_distance,
        math.atan2(dy, dx),
    )


class PoseController:
    """Generate skid-steer wheel commands from measured planar pose.

    Wheel order matches the project controller:
    front-left, front-right, rear-left, rear-right.  Equal wheel velocities
    drive forward; [-w, +w, -w, +w] rotates counter-clockwise.
    """

    def __init__(self, config: PoseControllerConfig | None = None):
        self.config = config or PoseControllerConfig()
        self.config.validate()
        self._stable_samples = 0
        self._position_acquired = False

    def reset(self) -> None:
        self._stable_samples = 0
        self._position_acquired = False

    def _drive_to_position(
        self,
        current: Pose2D,
        target: Pose2D,
        distance: float,
        target_yaw_error: float,
        allow_reverse: bool,
        phase_prefix: str = "",
    ) -> ControlOutput:
        """Generate the translation phase without final-yaw chatter."""
        cfg = self.config
        dx = target.x - current.x
        dy = target.y - current.y
        bearing = math.atan2(dy, dx)
        heading_error = wrap_angle(bearing - current.yaw)
        direction = 1.0
        if allow_reverse:
            reverse_error = wrap_angle(bearing + math.pi - current.yaw)
            if abs(reverse_error) < abs(heading_error):
                heading_error = reverse_error
                direction = -1.0
        angular = _clamp(
            cfg.angular_kp * heading_error,
            cfg.max_angular_wheel_speed,
        )
        angular = _signed_floor(angular, cfg.min_angular_wheel_speed)
        if abs(heading_error) >= cfg.rotate_in_place_threshold:
            linear = 0.0
            phase = "turn_to_path"
        else:
            linear = min(cfg.max_linear_wheel_speed, cfg.linear_kp * distance)
            linear = max(cfg.min_linear_wheel_speed, linear)
            linear *= max(0.0, math.cos(heading_error))
            linear *= direction
            phase = (
                "reverse_to_position"
                if direction < 0.0
                else "drive_to_position"
            )
        return ControlOutput(
            tuple(self._mix(linear, angular)),
            distance,
            target_yaw_error,
            False,
            phase_prefix + phase,
        )

    def _mix(self, linear: float, angular: float) -> tuple[float, ...]:
        yaw_term = angular * self.config.yaw_wheel_gain
        wheels = [
            linear - yaw_term,
            linear + yaw_term,
            linear - yaw_term,
            linear + yaw_term,
        ]
        peak = max(abs(value) for value in wheels)
        if peak > self.config.max_wheel_speed:
            scale = self.config.max_wheel_speed / peak
            wheels = [value * scale for value in wheels]
        return tuple(wheels)

    def compute_position(
        self,
        current: Pose2D,
        target: Pose2D,
        allow_reverse: bool = False,
        arrival_tolerance: float | None = None,
    ) -> ControlOutput:
        """Converge x/y only; final yaw belongs to a separate phase."""
        tolerance = (
            self.config.position_tolerance
            if arrival_tolerance is None
            else float(arrival_tolerance)
        )
        if tolerance <= 0.0:
            raise ValueError("position arrival tolerance must be positive")
        distance = math.hypot(target.x - current.x, target.y - current.y)
        target_yaw_error = wrap_angle(target.yaw - current.yaw)
        if distance > tolerance:
            self._stable_samples = 0
            return self._drive_to_position(
                current,
                target,
                distance,
                target_yaw_error,
                allow_reverse,
            )
        self._stable_samples += 1
        reached = self._stable_samples >= self.config.required_stable_samples
        return ControlOutput(
            (0.0, 0.0, 0.0, 0.0),
            distance,
            target_yaw_error,
            reached,
            "position_reached" if reached else "verify_position",
        )

    def compute_yaw(self, current: Pose2D, target: Pose2D) -> ControlOutput:
        """Converge yaw only; skid-steer translation drift is diagnostic."""
        distance = math.hypot(target.x - current.x, target.y - current.y)
        yaw_error = wrap_angle(target.yaw - current.yaw)
        if abs(yaw_error) > self.config.yaw_tolerance:
            self._stable_samples = 0
            angular = _clamp(
                self.config.angular_kp * yaw_error,
                self.config.max_angular_wheel_speed,
            )
            angular = _signed_floor(
                angular,
                self.config.min_angular_wheel_speed,
            )
            return ControlOutput(
                tuple(self._mix(0.0, angular)),
                distance,
                yaw_error,
                False,
                "turn_to_yaw",
            )
        self._stable_samples += 1
        reached = self._stable_samples >= self.config.required_stable_samples
        return ControlOutput(
            (0.0, 0.0, 0.0, 0.0),
            distance,
            yaw_error,
            reached,
            "yaw_reached" if reached else "verify_yaw",
        )

    def compute_straight(
        self,
        current: Pose2D,
        target: Pose2D,
        travel_yaw: float,
        arrival_tolerance: float,
    ) -> ControlOutput:
        """Drive on one fixed world heading without chasing endpoint bearing.

        This mode is for grasp, retreat, and placement corridors.  The ordinary position
        controller continually points at the remaining x/y vector; near the
        endpoint that vector is dominated by wheel slip and can reverse by
        180 degrees.  Here completion is based on longitudinal distance along
        the corridor, while yaw feedback keeps the chassis parallel to it.
        A negative longitudinal error naturally commands a straight reverse.
        """
        arrival_tolerance = float(arrival_tolerance)
        if arrival_tolerance <= 0.0:
            raise ValueError("arrival_tolerance must be positive")
        axis_x = math.cos(float(travel_yaw))
        axis_y = math.sin(float(travel_yaw))
        dx = target.x - current.x
        dy = target.y - current.y
        longitudinal_error = dx * axis_x + dy * axis_y
        yaw_error = wrap_angle(float(travel_yaw) - current.yaw)

        if abs(longitudinal_error) <= arrival_tolerance:
            self._stable_samples += 1
            reached = self._stable_samples >= self.config.required_stable_samples
            return ControlOutput(
                (0.0, 0.0, 0.0, 0.0),
                abs(longitudinal_error),
                yaw_error,
                reached,
                "straight_reached" if reached else "verify_straight",
            )

        self._stable_samples = 0
        linear = _clamp(
            self.config.linear_kp * longitudinal_error,
            self.config.max_linear_wheel_speed,
        )
        linear = _signed_floor(linear, self.config.min_linear_wheel_speed)
        # Do not apply the angular minimum during translation.  A forced
        # 0.30-rad/s correction for tiny yaw noise makes the loaded chassis
        # weave and was the source of the observed endpoint spiral.
        # Treat pose noise inside the accepted yaw tolerance as zero.  Without
        # this deadband the sign of a sub-degree error can alternate on every
        # sample, producing visible left-right wheel corrections even though
        # the chassis was already visually aligned.
        angular = 0.0
        if abs(yaw_error) > self.config.yaw_tolerance:
            angular = _clamp(
                self.config.angular_kp * yaw_error,
                self.config.max_angular_wheel_speed,
            )
        return ControlOutput(
            tuple(self._mix(linear, angular)),
            abs(longitudinal_error),
            yaw_error,
            False,
            "straight_reverse" if linear < 0.0 else "straight_forward",
        )

    def compute(
        self,
        current: Pose2D,
        target: Pose2D,
        allow_reverse: bool = False,
    ) -> ControlOutput:
        cfg = self.config
        dx = target.x - current.x
        dy = target.y - current.y
        distance = math.hypot(dx, dy)
        target_yaw_error = wrap_angle(target.yaw - current.yaw)

        # Once translation has reached its tight tolerance, keep the controller
        # in the final-yaw phase while in-place rotation causes only small pose
        # noise.  Without this hysteresis the loaded robot alternates at exactly
        # 0.010 m between path steering and final-yaw steering and never turns.
        if (
            self._position_acquired
            and distance > cfg.position_reacquire_tolerance
        ):
            self._position_acquired = False

        if not self._position_acquired and distance > cfg.position_tolerance:
            self._stable_samples = 0
            return self._drive_to_position(
                current,
                target,
                distance,
                target_yaw_error,
                allow_reverse,
            )

        self._position_acquired = True

        if abs(target_yaw_error) > cfg.yaw_tolerance:
            self._stable_samples = 0
            angular = _clamp(cfg.angular_kp * target_yaw_error,
                             cfg.max_angular_wheel_speed)
            angular = _signed_floor(angular, cfg.min_angular_wheel_speed)
            return ControlOutput(
                tuple(self._mix(0.0, angular)),
                distance,
                target_yaw_error,
                False,
                "turn_to_final_yaw",
            )

        # Rotation has converged, but a small translation drifted outside the
        # strict completion tolerance. Reacquire position now, then finish yaw
        # once more. This keeps the final reached contract exact.
        if distance > cfg.position_tolerance:
            self._position_acquired = False
            self._stable_samples = 0
            return self._drive_to_position(
                current,
                target,
                distance,
                target_yaw_error,
                allow_reverse,
                phase_prefix="reacquire_",
            )

        self._stable_samples += 1
        reached = self._stable_samples >= cfg.required_stable_samples
        return ControlOutput(
            (0.0, 0.0, 0.0, 0.0),
            distance,
            target_yaw_error,
            reached,
            "reached" if reached else "verify_settle",
        )
