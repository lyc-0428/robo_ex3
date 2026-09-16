"""URDF-based planar arm shifts without changing any finger command.

Only the three arm joints are solved.  The base remains fixed and the target
keeps the input tool pitch.  An unreachable request raises ArmReachError;
it is never silently clamped into a different grasp position.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np


ARM_JOINTS = ("arm_1_joint", "arm_2_joint", "endpoint_bracket_joint")


class ArmReachError(ValueError):
    """The requested fixed-base tool position cannot be reached accurately."""


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-12:
        raise ValueError("URDF contains a zero joint axis")
    x, y, z = axis / norm
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def _angle(value):
    return math.atan2(math.sin(value), math.cos(value))


class LinearArmKinematics:
    """Solve local x/z translations in the chassis frame.

    ``fk(pose, joint_names=None)`` returns ``[x, z, pitch]`` in metres/radians.
    ``shift(pose, joint_names=None, dx=0, dz=0)`` returns a complete pose list.
    Names may be supplied at construction instead.  All entries except the
    three ARM_JOINTS are preserved exactly, including tennis retaining pads.
    """

    def __init__(self, urdf_path, joint_names=None,
                 tip_link="endpoint_bracket_link", root_link="chassis_base_link"):
        self.joint_names = tuple(joint_names) if joint_names is not None else None
        joints = ET.parse(urdf_path).getroot().findall("joint")
        parents = {j.find("child").get("link"): j for j in joints}
        chain = []
        child = tip_link
        seen = set()
        while child != root_link:
            if child in seen or child not in parents:
                raise ValueError(f"No URDF chain from {root_link} to {tip_link}")
            seen.add(child)
            joint = parents[child]
            chain.append(joint)
            child = joint.find("parent").get("link")
        self.chain = []
        self.limits = {}
        for joint in reversed(chain):
            origin = joint.find("origin")
            xyz = [0.0] * 3 if origin is None else list(map(float, origin.get("xyz", "0 0 0").split()))
            rpy = [0.0] * 3 if origin is None else list(map(float, origin.get("rpy", "0 0 0").split()))
            matrix = np.eye(4)
            matrix[:3, 3] = xyz
            matrix[:3, :3] = (_rotation([0, 0, 1], rpy[2])
                              @ _rotation([0, 1, 0], rpy[1])
                              @ _rotation([1, 0, 0], rpy[0]))
            axis_node = joint.find("axis")
            axis = [1.0, 0.0, 0.0] if axis_node is None else list(map(float, axis_node.get("xyz", "1 0 0").split()))
            name, kind = joint.get("name"), joint.get("type")
            self.chain.append((name, kind, matrix, np.asarray(axis)))
            if name in ARM_JOINTS:
                limit = joint.find("limit")
                if kind == "revolute":
                    if limit is None or limit.get("lower") is None or limit.get("upper") is None:
                        raise ValueError(f"Missing limits for {name}")
                    self.limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
                elif kind == "continuous":
                    self.limits[name] = (-math.inf, math.inf)
                else:
                    raise ValueError(f"Unsupported arm joint type: {name}={kind}")
        if set(self.limits) != set(ARM_JOINTS):
            raise ValueError("Tool chain must include all three arm joints")

    def _inputs(self, pose, joint_names):
        names = tuple(joint_names) if joint_names is not None else self.joint_names
        if names is None or len(names) != len(pose) or len(set(names)) != len(names):
            raise ValueError("A unique joint name is required for each pose entry")
        if not set(ARM_JOINTS).issubset(names):
            raise ValueError("Pose is missing one or more arm joints")
        values = np.asarray(pose, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("Pose entries must be finite scalars")
        return names, values

    def _fk(self, values):
        result = np.eye(4)
        for name, kind, origin, axis in self.chain:
            motion = np.eye(4)
            q = float(values.get(name, 0.0))
            if kind in ("revolute", "continuous"):
                motion[:3, :3] = _rotation(axis, q)
            elif kind == "prismatic":
                motion[:3, 3] = axis * q
            result = result @ origin @ motion
        # All three arm axes in this model are parallel to chassis +y.
        pitch = math.atan2(result[0, 2], result[0, 0])
        return np.array([result[0, 3], result[2, 3], pitch])

    def fk(self, pose, joint_names=None):
        names, values = self._inputs(pose, joint_names)
        return self._fk(dict(zip(names, values)))

    def shift(self, pose, joint_names=None, dx=0.0, dz=0.0):
        names, values = self._inputs(pose, joint_names)
        if not math.isfinite(dx) or not math.isfinite(dz):
            raise ValueError("Tool shifts must be finite")
        indices = [names.index(name) for name in ARM_JOINTS]
        original = values[indices].copy()
        lower = np.array([self.limits[n][0] for n in ARM_JOINTS])
        upper = np.array([self.limits[n][1] for n in ARM_JOINTS])
        if np.any(original < lower - 1.0e-6) or np.any(original > upper + 1.0e-6):
            raise ArmReachError("Input arm command already exceeds its URDF limits")
        # Avoid a whole revolution in a continuous joint when retracting a load.
        lower = np.maximum(lower, original - math.pi)
        upper = np.minimum(upper, original + math.pi)
        base = dict(zip(names, values))
        target = self._fk(base) + np.array([dx, dz, 0.0])
        weights = np.array([1.0, 1.0, 0.10])

        def evaluate(q):
            current = self._fk(dict(base, **dict(zip(ARM_JOINTS, q))))
            error = target - current
            error[2] = _angle(float(error[2]))
            return error

        best_error = np.array([math.inf, math.inf, math.inf])
        # The current elbow branch is preferred. Alternative seeds help when
        # the initial extended posture is close to a singularity.
        for offset in ((0, 0, 0), (-0.35, 0.55, -0.20), (0.20, -0.40, 0.20)):
            q = np.clip(original + np.array(offset), lower, upper)
            for _ in range(100):
                error = evaluate(q)
                if np.linalg.norm(error * weights) < np.linalg.norm(best_error * weights):
                    best_error = error.copy()
                if np.linalg.norm(error[:2]) <= 0.0003 and abs(error[2]) <= 0.003:
                    result = list(pose)
                    for index, value in zip(indices, q):
                        result[index] = float(value)
                    return result
                jacobian = np.empty((3, 3))
                epsilon = 1.0e-5
                for column in range(3):
                    perturbed = q.copy()
                    perturbed[column] += epsilon
                    difference = evaluate(perturbed) - error
                    difference[2] = _angle(float(difference[2]))
                    jacobian[:, column] = -difference / epsilon
                scaled = weights[:, None] * jacobian
                delta = scaled.T @ np.linalg.solve(
                    scaled @ scaled.T + 1.0e-6 * np.eye(3), error * weights)
                delta *= min(1.0, 0.20 / max(float(np.max(np.abs(delta))), 1.0e-12))
                improved = False
                for fraction in (1.0, 0.5, 0.25, 0.10):
                    candidate = np.clip(q + fraction * delta, lower, upper)
                    if np.linalg.norm(evaluate(candidate) * weights) < np.linalg.norm(error * weights) - 1.0e-10:
                        q = candidate
                        improved = True
                        break
                if not improved:
                    break
        raise ArmReachError(
            f"Unreachable arm shift dx={dx:.4f} m dz={dz:.4f} m; "
            f"best position error={np.linalg.norm(best_error[:2]):.4f} m, "
            f"pitch error={math.degrees(abs(best_error[2])):.2f} deg")
