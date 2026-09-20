"""Closed-loop heading options that do not use calibrated turn duration."""

from __future__ import annotations

from dataclasses import dataclass
import math


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class YawPid:
    kp: float = 3.0
    ki: float = 0.05
    kd: float = 0.15
    maximum: float = 2.0
    integral_limit: float = 0.8
    _integral: float = 0.0
    _previous_error: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._previous_error = None

    def update(self, target_yaw: float, measured_yaw: float, dt: float) -> float:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        error = wrap(target_yaw - measured_yaw)
        self._integral = max(
            -self.integral_limit,
            min(self.integral_limit, self._integral + error * dt),
        )
        derivative = 0.0
        if self._previous_error is not None:
            derivative = wrap(error - self._previous_error) / dt
        self._previous_error = error
        command = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(-self.maximum, min(self.maximum, command))


def visual_servo_turn(error_pixels: float, kp=0.02,
                      deadband=12.0, maximum=1.8) -> float:
    """Pixel feedback is useful for the final camera-centering phase."""
    if abs(error_pixels) <= deadband:
        return 0.0
    return max(-maximum, min(maximum, -kp * error_pixels))


def bearing_to(x_from: float, y_from: float,
               x_to: float, y_to: float) -> float:
    return math.atan2(y_to - y_from, x_to - x_from)

