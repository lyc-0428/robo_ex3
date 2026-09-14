#!/usr/bin/env python3
"""Pure helpers shared by the simulated scene, detector, and sorter."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import random
import re


BOTTLE = "bottle"
TENNIS = "tennis"
UNKNOWN = "unknown"
TENNIS_COLLISION_SIDE = 0.054
TENNIS_VISUAL_SCALE = 0.93
TENNIS_SPAWN_HEIGHT = TENNIS_COLLISION_SIDE * 0.5

_BOTTLE_ALIASES = {
    "bottle",
    "water bottle",
    "waterbottle",
}
_TENNIS_ALIASES = {
    "ball",
    "sports ball",
    "tennis",
    "tennis ball",
    "tennisball",
}


def normalize_label(value):
    """Normalize YOLO labels without hiding the model's original names."""
    text = str(value).strip().lower()
    text = re.sub(r"[_-]+", " ", text)
    return " ".join(text.split())


def canonical_class(value):
    """Map a supported model label to the two task classes."""
    label = normalize_label(value)
    if label in _BOTTLE_ALIASES:
        return BOTTLE
    if label in _TENNIS_ALIASES:
        return TENNIS
    return None


def validate_model_names(names):
    """Return class-id -> task-class and fail if either class is absent."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    mapping = {}
    found = set()
    actual = []
    for class_id, raw_name in items:
        actual.append(str(raw_name))
        task_class = canonical_class(raw_name)
        if task_class is not None:
            mapping[int(class_id)] = task_class
            found.add(task_class)
    missing = [name for name in (BOTTLE, TENNIS) if name not in found]
    if missing:
        raise ValueError(
            "YOLO model is missing required task classes "
            f"{missing}; actual classes: {actual}"
        )
    return mapping


def bbox_center_x(detection):
    bbox = detection["bbox"]
    return (float(bbox["x1"]) + float(bbox["x2"])) * 0.5


def _frame_candidate(
    detections, principal_x, preferred_class=None, selection="center"
):
    candidates = []
    for detection in detections:
        task_class = canonical_class(detection.get("class_name", ""))
        if task_class is None:
            continue
        if preferred_class is not None and task_class != preferred_class:
            continue
        item = dict(detection)
        item["class_name"] = task_class
        candidates.append(item)
    if not candidates:
        return None
    if selection == "leftmost":
        return min(
            candidates,
            key=lambda item: (
                bbox_center_x(item),
                -float(item.get("confidence", 0.0)),
            ),
        )
    if selection != "center":
        raise ValueError(f"unsupported target selection: {selection}")
    return min(
        candidates,
        key=lambda item: (
            abs(bbox_center_x(item) - float(principal_x)),
            -float(item.get("confidence", 0.0)),
        ),
    )


def stable_center_detection(
    frames,
    principal_x,
    required_votes=3,
    preferred_class=None,
    max_center_spread_px=90.0,
    selection="center",
):
    """Select a center target that is stable across recent detector frames."""
    per_frame = []
    for detections in frames:
        candidate = _frame_candidate(
            detections,
            principal_x,
            preferred_class=preferred_class,
            selection=selection,
        )
        if candidate is not None:
            per_frame.append(candidate)
    if len(per_frame) < int(required_votes):
        return None

    votes = Counter(item["class_name"] for item in per_frame)
    task_class, vote_count = votes.most_common(1)[0]
    if vote_count < int(required_votes):
        return None

    matching = [item for item in per_frame if item["class_name"] == task_class]
    centers = [bbox_center_x(item) for item in matching]
    if max(centers) - min(centers) > float(max_center_spread_px):
        return None

    result = dict(max(matching, key=lambda item: float(item.get("confidence", 0.0))))
    result["class_name"] = task_class
    result["confidence"] = sum(
        float(item.get("confidence", 0.0)) for item in matching
    ) / len(matching)
    result["bbox"] = {
        key: sum(float(item["bbox"][key]) for item in matching) / len(matching)
        for key in ("x1", "y1", "x2", "y2")
    }
    result["stable_votes"] = len(matching)
    return result


def horizontal_angle_radians(detection, focal_x, principal_x):
    """Return a positive angle for a target to the image/right side."""
    focal_x = float(focal_x)
    if focal_x <= 0.0:
        raise ValueError("focal_x must be positive")
    pixel_offset = bbox_center_x(detection) - float(principal_x)
    return math.atan2(pixel_offset, focal_x)


def angle_to_steps(angle_radians, steps_per_radian):
    if float(steps_per_radian) <= 0.0:
        raise ValueError("steps_per_radian must be positive")
    return int(round(float(angle_radians) * float(steps_per_radian)))


def placement_step(sequence, completed_for_class):
    values = [int(value) for value in sequence]
    if not values or any(value <= 0 for value in values):
        raise ValueError("placement step sequence must contain positive values")
    index = min(int(completed_for_class), len(values) - 1)
    return values[index]


def make_slot_layout(seed, radius, angles_degrees, min_separation=0.12):
    """Create a deterministic, randomized, evenly balanced task scene.

    Entity names stay class-neutral.  The seed now materially changes which
    class occupies each slot, preventing the controller from relying on a
    memorized left-to-right class pattern.
    """
    radius = float(radius)
    angles = [float(value) for value in angles_degrees]
    if radius <= 0.0 or len(angles) < 2 or len(angles) % 2:
        raise ValueError("slot radius must be positive and slot count must be even")
    if len(set(angles)) != len(angles):
        raise ValueError("slot angles must be unique")
    classes = [BOTTLE] * (len(angles) // 2) + [TENNIS] * (len(angles) // 2)
    random.Random(int(seed)).shuffle(classes)
    layout = []
    for index, (angle_degrees, task_class) in enumerate(zip(angles, classes)):
        angle = math.radians(angle_degrees)
        layout.append(
            {
                "name": f"task_object_{index}",
                "class_name": task_class,
                "angle_degrees": angle_degrees,
                "x": radius * math.cos(angle),
                "y": radius * math.sin(angle),
                # Bottle collision is ground-referenced internally.  The
                # tennis cube collision is centred on the model origin.
                "z": 0.0 if task_class == BOTTLE else TENNIS_SPAWN_HEIGHT,
            }
        )
    pair_distances = [
        math.hypot(right["x"] - left["x"], right["y"] - left["y"])
        for left, right in zip(layout, layout[1:])
    ]
    if pair_distances and min(pair_distances) < float(min_separation):
        raise ValueError(
            f"adjacent slot separation {min(pair_distances):.3f} m is below "
            f"the required {float(min_separation):.3f} m"
        )
    return layout


def _resource_uri(model_root, relative_path):
    path = (Path(model_root) / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"model resource not found: {path}")
    return path.as_uri()


def _surface_sdf(friction):
    return f"""<surface>
          <friction><ode><mu>{friction:.4f}</mu><mu2>{friction:.4f}</mu2></ode></friction>
          <contact><ode><kp>200000</kp><kd>80</kd><max_vel>0.05</max_vel><min_depth>0.001</min_depth></ode></contact>
        </surface>"""


def _hex_points(circumradius):
    return [
        (
            circumradius * math.cos(math.radians(30.0 + 60.0 * index)),
            circumradius * math.sin(math.radians(30.0 + 60.0 * index)),
        )
        for index in range(6)
    ]


def bottle_sdf(name, model_root, mass=0.20, friction=1.30):
    """Bottle visual with a convex regular-hexagonal-prism collision body."""
    _resource_uri(model_root, "water_bottle_03/meshes/bottle_complete.obj")
    radius = 0.034
    height = 0.200
    mass = float(mass)
    friction = float(friction)
    transverse_inertia = mass * (3.0 * radius**2 + height**2) / 12.0
    axial_inertia = 0.5 * mass * radius**2
    points = "\n".join(
        f"            <point>{x:.9f} {y:.9f}</point>"
        for x, y in _hex_points(radius)
    )
    return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <static>false</static>
    <link name="object_link">
      <!-- SDF polyline extrusion starts at local z=0 and grows upward. -->
      <pose>0 0 0 0 0 0</pose>
      <gravity>true</gravity>
      <velocity_decay><linear>0.05</linear><angular>0.10</angular></velocity_decay>
      <inertial>
        <pose>0 0 {height * 0.5:.6f} 0 0 0</pose>
        <mass>{mass:.9f}</mass>
        <inertia>
          <ixx>{transverse_inertia:.12f}</ixx><iyy>{transverse_inertia:.12f}</iyy>
          <izz>{axial_inertia:.12f}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="easy_grip_hex_prism_collision">
        <geometry><polyline><height>{height:.6f}</height>
{points}
        </polyline></geometry>
        {_surface_sdf(friction)}
      </collision>
      <visual name="bottle_visual">
        <!-- Mesh minimum z is 0.003398778 m after scaling. -->
        <pose>0 0 -0.003398778 0 0 0</pose>
        <geometry><mesh>
          <uri>model://water_bottle_03/meshes/bottle_complete.obj</uri>
          <scale>0.06706608 0.06706608 0.05761422</scale>
        </mesh></geometry>
        <!-- Pale blue is both human-visible and usable as detector evidence. -->
        <material>
          <ambient>0.32 0.52 0.72 1</ambient>
          <diffuse>0.48 0.72 0.92 1</diffuse>
          <specular>0.24 0.30 0.36 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def tennis_sdf(name, model_root, mass=0.057, friction=1.60):
    """Tennis visual with a flat, easy-to-grip cubic collision body."""
    side = TENNIS_COLLISION_SIDE
    visual_scale = TENNIS_VISUAL_SCALE
    mass = float(mass)
    friction = float(friction)
    inertia = mass * side**2 / 6.0
    mesh_uri = _resource_uri(model_root, "056_tennis_ball/textured.obj")
    return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <static>false</static>
    <link name="object_link">
      <gravity>true</gravity>
      <velocity_decay>
        <linear>0.08</linear>
        <angular>0.15</angular>
      </velocity_decay>
      <inertial>
        <mass>{mass:.9f}</mass>
        <inertia>
          <ixx>{inertia:.12f}</ixx>
          <iyy>{inertia:.12f}</iyy>
          <izz>{inertia:.12f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="object_collision">
        <!-- The visual remains a tennis ball; only contact geometry is cubic. -->
        <geometry>
          <box><size>{side:.6f} {side:.6f} {side:.6f}</size></box>
        </geometry>
        {_surface_sdf(friction)}
      </collision>
      <visual name="object_visual">
        <pose>-0.007637 0.041179 -0.030812 0 0 0</pose>
        <geometry><mesh>
          <uri>{mesh_uri}</uri>
          <scale>{visual_scale:.6f} {visual_scale:.6f} {visual_scale:.6f}</scale>
        </mesh></geometry>
      </visual>
    </link>
  </model>
</sdf>
"""


def unknown_sdf(name, mass=0.08, friction=1.20):
    """Neutral gray object used only for the unrecognized-object test."""
    side = 0.060
    height = 0.090
    ixx = mass * (side**2 + height**2) / 12.0
    izz = mass * (side**2 + side**2) / 12.0
    return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <static>false</static>
    <link name="object_link">
      <gravity>true</gravity>
      <inertial><mass>{mass:.9f}</mass><inertia>
        <ixx>{ixx:.12f}</ixx><iyy>{ixx:.12f}</iyy><izz>{izz:.12f}</izz>
        <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
      </inertia></inertial>
      <collision name="unknown_box_collision">
        <geometry><box><size>{side:.6f} {side:.6f} {height:.6f}</size></box></geometry>
        {_surface_sdf(float(friction))}
      </collision>
      <visual name="unknown_gray_visual">
        <geometry><box><size>{side:.6f} {side:.6f} {height:.6f}</size></box></geometry>
        <material><ambient>0.35 0.35 0.35 1</ambient><diffuse>0.42 0.42 0.42 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
"""
