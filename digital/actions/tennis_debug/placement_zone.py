"""Pure geometry helpers for broad left/right classification regions."""

from __future__ import annotations

import math


def lateral_offset(
    origin_x,
    origin_y,
    origin_yaw,
    object_x,
    object_y,
):
    """Return object displacement along the robot's initial left axis."""
    dx = float(object_x) - float(origin_x)
    dy = float(object_y) - float(origin_y)
    yaw = float(origin_yaw)
    return -math.sin(yaw) * dx + math.cos(yaw) * dy


def in_classification_half(
    origin_x,
    origin_y,
    origin_yaw,
    object_x,
    object_y,
    *,
    left_half,
    margin,
):
    """Check a broad class half-plane with a margin from the centre line."""
    margin = float(margin)
    if margin <= 0.0:
        raise ValueError("classification margin must be positive")
    offset = lateral_offset(
        origin_x,
        origin_y,
        origin_yaw,
        object_x,
        object_y,
    )
    return offset >= margin if left_half else offset <= -margin


def placement_distance_for_attempt(start, decrement, minimum, attempt):
    """Return a one-based, monotonically decreasing placement distance."""
    start = float(start)
    decrement = float(decrement)
    minimum = float(minimum)
    attempt = int(attempt)
    if min(start, minimum) <= 0.0:
        raise ValueError("placement distances must be positive")
    if decrement < 0.0:
        raise ValueError("placement decrement cannot be negative")
    if minimum > start:
        raise ValueError("minimum placement distance exceeds start")
    if attempt < 1:
        raise ValueError("attempt must be one-based")
    return max(minimum, start - (attempt - 1) * decrement)
