"""Pure helpers for the staged tennis-ball gripper motion."""

from __future__ import annotations


LEFT_ROOT = "left_gripper_joint_1"
RIGHT_ROOT = "right_gripper_joint_1"
LEFT_TIP = "left_gripper_joint_5"
RIGHT_TIP = "right_gripper_joint_5"


def tennis_closed_pose(pose, initial, index, tip_offset):
    """Return a closed pose whose front finger links curl toward the centre.

    The two fingers are mirrored about the gripper centreline.  With the URDF
    joint axes, decreasing left joint 5 and increasing right joint 5 moves the
    front pads inward.  Root targets are supplied by the ordinary gripper
    pose so existing grasp calibration remains authoritative.
    """
    if tip_offset < 0.0:
        raise ValueError("tip_offset cannot be negative")
    result = list(pose)
    result[index[LEFT_TIP]] = initial[index[LEFT_TIP]] - tip_offset
    result[index[RIGHT_TIP]] = initial[index[RIGHT_TIP]] + tip_offset
    return result


def tennis_open_pose(pose, initial, index):
    """Restore both front-pad joints to calibration for a clean release."""
    result = list(pose)
    result[index[LEFT_TIP]] = initial[index[LEFT_TIP]]
    result[index[RIGHT_TIP]] = initial[index[RIGHT_TIP]]
    return result


def tip_first_pose(start, closed, index):
    """Move only the two front-pad joints, leaving both roots fully open."""
    result = list(start)
    result[index[LEFT_TIP]] = closed[index[LEFT_TIP]]
    result[index[RIGHT_TIP]] = closed[index[RIGHT_TIP]]
    return result


def is_tip_first_closing(start, target, index, tolerance=1.0e-6):
    """Recognize the tennis-only closing command, never open/release moves."""
    return (
        target[index[LEFT_TIP]] < start[index[LEFT_TIP]] - tolerance
        and target[index[RIGHT_TIP]] > start[index[RIGHT_TIP]] + tolerance
    )
