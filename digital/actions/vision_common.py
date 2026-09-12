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


def make_slot_layout(seed, radius, angles_degrees):
    """Create neutral-name slots; the returned order never encodes class."""
    radius = float(radius)
    angles = [float(value) for value in angles_degrees]
    if radius <= 0.0 or len(angles) != 4:
        raise ValueError("slot radius must be positive and exactly four angles are required")
    classes = [BOTTLE] * 2 + [TENNIS] * 2
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
                # water_bottle_01 places its link centre at z=0.105 inside
                # the model, so its model origin belongs directly on z=0.
                "z": 0.0 if task_class == BOTTLE else 0.0335,
            }
        )
    return layout


def _resource_uri(model_root, relative_path):
    path = (Path(model_root) / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"model resource not found: {path}")
    return path.as_uri()


def bottle_sdf(name, model_root):
    """Load the team-authored bottle SDF without changing its geometry."""
    model_path = Path(model_root) / "water_bottle_01" / "model.sdf"
    if not model_path.is_file():
        raise FileNotFoundError(f"custom bottle model not found: {model_path}")
    source = model_path.read_text(encoding="utf-8")
    original_tag = '<model name="water_bottle_01">'
    if source.count(original_tag) != 1:
        raise ValueError(
            "water_bottle_01/model.sdf must contain exactly one expected model tag"
        )
    # Neutral naming is the only modification. All link poses, mesh URIs,
    # materials, mass, collision dimensions and inertias remain verbatim.
    return source.replace(original_tag, f'<model name="{name}">', 1)


def tennis_sdf(name, model_root, mass=0.057, friction=5.0):
    radius = 0.0335
    mass = float(mass)
    friction = float(friction)
    inertia = 0.4 * mass * radius**2
    mesh_uri = _resource_uri(model_root, "056_tennis_ball/textured.obj")
    return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <static>false</static>
    <link name="object_link">
      <gravity>false</gravity>
      <inertial>
        <mass>{mass:.9f}</mass>
        <inertia>
          <ixx>{inertia:.12f}</ixx><iyy>{inertia:.12f}</iyy><izz>{inertia:.12f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="object_collision">
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
        <surface>
          <friction><ode><mu>{friction}</mu><mu2>{friction}</mu2></ode></friction>
          <contact><ode><kp>100000</kp><kd>10</kd></ode></contact>
        </surface>
      </collision>
      <visual name="object_visual">
        <pose>-0.0082115 0.044278 -0.0331315 0 0 0</pose>
        <geometry><mesh><uri>{mesh_uri}</uri></mesh></geometry>
      </visual>
    </link>
  </model>
</sdf>
"""
