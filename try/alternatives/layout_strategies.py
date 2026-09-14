"""Three deterministic layouts for six fixed pickup cells."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class Slot:
    index: int
    x: float
    y: float


def minimum_separation(slots: list[Slot]) -> float:
    if len(slots) < 2:
        return float("inf")
    return min(
        math.hypot(a.x - b.x, a.y - b.y)
        for number, a in enumerate(slots)
        for b in slots[number + 1:]
    )


def validate(slots: list[Slot], robot_clearance=0.40, spacing=0.12) -> None:
    if len({slot.index for slot in slots}) != len(slots):
        raise ValueError("slot indices must be unique")
    if any(math.hypot(slot.x, slot.y) < robot_clearance for slot in slots):
        raise ValueError("an object is too close to the chassis")
    if minimum_separation(slots) < spacing:
        raise ValueError("objects are too close to one another")


def wide_arc(radius=0.47, angles=(-40, -24, -8, 8, 24, 40)) -> list[Slot]:
    slots = [
        Slot(index, radius * math.cos(math.radians(angle)),
             radius * math.sin(math.radians(angle)))
        for index, angle in enumerate(angles)
    ]
    validate(slots)
    return slots


def staggered_rows() -> list[Slot]:
    """Two rows keep camera boxes separate but require forward translation."""
    coordinates = (
        (0.43, -0.24), (0.43, 0.00), (0.43, 0.24),
        (0.56, -0.18), (0.56, 0.06), (0.56, 0.30),
    )
    slots = [Slot(index, x, y) for index, (x, y) in enumerate(coordinates)]
    validate(slots)
    return slots


def two_arcs() -> list[Slot]:
    """A near/far fan increases spacing when the camera has enough depth."""
    polar = ((0.43, -36), (0.55, -24), (0.43, -8),
             (0.55, 8), (0.43, 24), (0.55, 36))
    slots = [
        Slot(index, radius * math.cos(math.radians(angle)),
             radius * math.sin(math.radians(angle)))
        for index, (radius, angle) in enumerate(polar)
    ]
    validate(slots)
    return slots


def randomized_balanced_classes(seed: int, count=6) -> list[str]:
    if count < 2 or count % 2:
        raise ValueError("balanced class count must be positive and even")
    values = ["bottle"] * (count // 2) + ["tennis"] * (count // 2)
    random.Random(seed).shuffle(values)
    return values

