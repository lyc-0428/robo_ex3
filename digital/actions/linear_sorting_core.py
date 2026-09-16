"""ROS-free geometry, visual target selection, and verified counting."""
from dataclasses import dataclass, field
import math
import random

ROW_X = 0.28005  # stable tennis standoff .300 minus calibrated .01995 insertion
SPACING = 0.30
LEFT_END = 0.90
RIGHT_END = -0.90
GRID_CENTERS = tuple((3-index)*SPACING for index in range(7))
EMPTY_GRID_INDEX = 3
EMPTY_GRID_CELL = "G4"
EMPTY_GRID_Y = GRID_CENTERS[EMPTY_GRID_INDEX]


def make_linear_layout(seed, row_x=ROW_X, spacing=SPACING):
    if not math.isfinite(row_x) or row_x <= 0 or spacing < 0.25:
        raise ValueError("row_x must be positive; spacing must be at least .25 m")
    classes = ["tennis"] * 3 + ["bottle"] * 3
    random.Random(int(seed)).shuffle(classes)
    occupied_centers = [center for index, center in enumerate(GRID_CENTERS)
                        if index != EMPTY_GRID_INDEX]
    return [dict(name=f"task_object_{i}", class_name=kind,
                 x=row_x, y=occupied_centers[i])
            for i, kind in enumerate(classes)]


def fixed_grid_cell(world_y):
    """Return the fixed G1..G7 table cell nearest a detected entity."""
    value = float(world_y)
    if not math.isfinite(value) or not RIGHT_END <= value <= LEFT_END:
        raise ValueError(f"target y={value} is outside the fixed table grid")
    index = min(range(len(GRID_CENTERS)),
                key=lambda item: abs(value-GRID_CENTERS[item]))
    center = GRID_CENTERS[index]
    return f"G{index+1}", center


def drop_y(kind, already_sorted):
    if kind not in ("tennis", "bottle") or not 0 <= already_sorted < 3:
        raise ValueError("invalid class or full placement area")
    return (1 if kind == "tennis" else -1) * (1.30 + .30*already_sorted)


def select_visible(frames, positions, completed, attempted, robot_y, direction,
                   band=.10, votes=2):
    """Require current same-ID/same-class detections, not pooled class votes."""
    if not frames:
        return None
    candidates = []
    for latest in frames[-1]:
        name, kind = latest.get("object_id"), latest.get("class_name")
        if name is None or kind not in ("tennis", "bottle"):
            continue
        if name in completed or name in attempted or name not in positions:
            continue
        position = positions[name]
        if abs(float(position[1])-robot_y) > band:
            continue
        matching = [next((d for d in frame if d.get("object_id") == name
                         and d.get("class_name") == kind), None) for frame in frames[-5:]]
        matching = [d for d in matching if d is not None]
        if len(matching) < votes:
            continue
        centers = [(d["bbox"]["x1"]+d["bbox"]["x2"])*.5 for d in matching]
        if max(centers)-min(centers) > 45:
            continue
        candidates.append(dict(latest))
    if not candidates:
        return None
    # Physical left-to-right order; world poses supply geometry, never class.
    return min(candidates, key=lambda d: direction*float(positions[d["object_id"]][1]))


@dataclass
class VerifiedLedger:
    completed: set = field(default_factory=set)
    counts: dict = field(default_factory=lambda: {"tennis": 0, "bottle": 0})

    def record(self, name, kind, origin, released, expected_y):
        if name in self.completed:
            return False
        if kind not in self.counts or self.counts[kind] >= 3:
            raise ValueError("unexpected class count")
        if not all(math.isfinite(float(v)) for v in (*origin, *released)):
            raise ValueError("invalid object pose")
        side = 1 if kind == "tennis" else -1
        if (side*released[1] < 1.10 or abs(released[1]-expected_y) > .10
                or abs(released[0]-ROW_X) > .10
                or math.hypot(released[0]-origin[0], released[1]-origin[1]) < .20):
            raise ValueError("object did not arrive in the intended drop slot")
        ground_z = .027 if kind == "tennis" else 0.0
        if abs(released[2]-ground_z) > .025:
            raise ValueError("object is not resting on the ground")
        self.completed.add(name)
        self.counts[kind] += 1
        return True

    @property
    def done(self):
        return len(self.completed) == 6 and self.counts == {"tennis": 3, "bottle": 3}


def rail_command(x, y, yaw, target_y, max_speed):
    """Body-frame velocity: world lateral travel with small rail/yaw corrections."""
    wrap_yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    if abs(x) > .035 or abs(wrap_yaw) > math.radians(5):
        raise ValueError("base departed fixed x/yaw rail")
    vx = max(-.015, min(.015, -2*x))
    vy = max(-max_speed, min(max_speed, 1.4*(target_y-y)))
    omega = max(-.08, min(.08, -2*wrap_yaw))
    return (math.cos(yaw)*vx + math.sin(yaw)*vy,
            -math.sin(yaw)*vx + math.cos(yaw)*vy, omega)
