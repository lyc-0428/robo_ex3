"""Alternative easy-grip collision geometries and inertia calculations."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Inertia:
    ixx: float
    iyy: float
    izz: float


def box_inertia(mass: float, x: float, y: float, z: float) -> Inertia:
    if min(mass, x, y, z) <= 0.0:
        raise ValueError("mass and box dimensions must be positive")
    return Inertia(
        mass * (y * y + z * z) / 12.0,
        mass * (x * x + z * z) / 12.0,
        mass * (x * x + y * y) / 12.0,
    )


def prism_inertia_approx(mass: float, radius: float, height: float) -> Inertia:
    """Conservative cylinder-equivalent inertia for a regular prism."""
    if min(mass, radius, height) <= 0.0:
        raise ValueError("mass, radius and height must be positive")
    transverse = mass * (3.0 * radius * radius + height * height) / 12.0
    axial = 0.5 * mass * radius * radius
    return Inertia(transverse, transverse, axial)


def regular_prism_polyline(sides: int, radius: float, height: float) -> str:
    """Return an SDFormat polyline for a convex n-sided vertical prism."""
    if sides < 3 or min(radius, height) <= 0.0:
        raise ValueError("a prism needs at least three sides and positive size")
    points = []
    phase = math.pi / sides
    for index in range(sides):
        angle = phase + index * 2.0 * math.pi / sides
        points.append(
            f"<point>{radius * math.cos(angle):.9f} "
            f"{radius * math.sin(angle):.9f}</point>"
        )
    return f"<polyline><height>{height:.6f}</height>{''.join(points)}</polyline>"


def collision_geometry(strategy: str, width: float, height: float) -> str:
    """Return one of three practical collision options.

    ``box`` is the most stable and easiest to tune. ``hex`` is the recommended
    compromise for bottle-like visuals. ``octagon`` is closer to a cylinder
    while still offering flat gripper faces.
    """
    strategy = strategy.strip().lower()
    if min(width, height) <= 0.0:
        raise ValueError("collision dimensions must be positive")
    if strategy == "box":
        return (
            f"<box><size>{width:.6f} {width:.6f} "
            f"{height:.6f}</size></box>"
        )
    if strategy == "hex":
        return regular_prism_polyline(6, width * 0.5, height)
    if strategy == "octagon":
        return regular_prism_polyline(8, width * 0.5, height)
    raise ValueError(f"unsupported collision strategy: {strategy}")


def contact_surface(friction: float = 1.3) -> str:
    if not 0.1 <= friction <= 3.0:
        raise ValueError("use a plausible ODE friction coefficient in [0.1, 3.0]")
    return (
        "<surface><friction><ode>"
        f"<mu>{friction:.4f}</mu><mu2>{friction:.4f}</mu2>"
        "</ode></friction><contact><ode>"
        "<kp>200000</kp><kd>80</kd><max_vel>0.05</max_vel>"
        "<min_depth>0.001</min_depth></ode></contact></surface>"
    )

