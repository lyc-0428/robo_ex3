#!/usr/bin/env python3
"""Independent HSV colour evidence for the simulated light-blue bottle."""

from __future__ import annotations

import cv2
import numpy as np


# OpenCV HSV range:
# H: 0..179
# S: 0..255
# V: 0..255
#
# The Gazebo bottle is a transparent pale cyan-blue object.
# Its rendered body is typically around H ~= 100, while transparency
# lowers saturation considerably.  Therefore saturation must not be
# constrained as aggressively as for an opaque blue object.
HSV_H_MIN = 75
HSV_H_MAX = 140
HSV_S_MIN = 8
HSV_S_MAX = 255
HSV_V_MIN = 40
HSV_V_MAX = 255
def bottle_hsv_mask(frame):
    """Return a binary mask for the transparent light-blue bottle colour."""

    image = np.asarray(frame)

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("frame must be an HxWx3 BGR image")

    bgr = image[:, :, :3]

    hsv = cv2.cvtColor(
        np.ascontiguousarray(bgr, dtype=np.uint8),
        cv2.COLOR_BGR2HSV,
    )

    lower = np.array(
        [HSV_H_MIN, HSV_S_MIN, HSV_V_MIN],
        dtype=np.uint8,
    )

    upper = np.array(
        [HSV_H_MAX, HSV_S_MAX, HSV_V_MAX],
        dtype=np.uint8,
    )

    # Light / ordinary blue.
    light_blue_mask = cv2.inRange(
        hsv,
        lower,
        upper,
    )

    # ---------------------------------------------------------
    # DARK BLUE SUPPORT
    #
    # Deep-blue bottle regions can become very dark under
    # Gazebo lighting. Their HSV hue is less stable, so use
    # both a wider dark-HSV range and direct BGR blue dominance.
    # ---------------------------------------------------------
    dark_hsv_lower = np.array(
        [88, 25, 15],
        dtype=np.uint8,
    )
    dark_hsv_upper = np.array(
        [165, 255, 190],
        dtype=np.uint8,
    )

    dark_hsv_mask = cv2.inRange(
        hsv,
        dark_hsv_lower,
        dark_hsv_upper,
    )

    b = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    r = image[:, :, 2].astype(np.int16)

    dark_bgr_mask = (
        (b >= 35)
        & (b >= g + 8)
        & (b >= r + 18)
    ).astype(np.uint8) * 255

    mask = cv2.bitwise_or(
        light_blue_mask,
        dark_hsv_mask,
    )

    mask = cv2.bitwise_or(
        mask,
        dark_bgr_mask,
    )

    # Remove isolated camera/rendering noise.
    kernel = np.ones((3, 3), dtype=np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    # Join fragmented transparent bottle pixels.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    return mask


def hsv_fraction_in_bbox(frame, bbox):
    """Return bottle-HSV fraction inside one bbox."""

    mask = bottle_hsv_mask(frame)

    height, width = mask.shape[:2]

    x1 = max(
        0,
        min(
            width,
            int(round(float(bbox["x1"]))),
        ),
    )

    y1 = max(
        0,
        min(
            height,
            int(round(float(bbox["y1"]))),
        ),
    )

    x2 = max(
        0,
        min(
            width,
            int(round(float(bbox["x2"]))),
        ),
    )

    y2 = max(
        0,
        min(
            height,
            int(round(float(bbox["y2"]))),
        ),
    )

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = mask[y1:y2, x1:x2]

    if roi.size == 0:
        return 0.0

    return float(np.mean(roi > 0))


def adjusted_bottle_confidence(
    frame,
    bbox,
    raw_confidence,
    bonus=0.60,
    minimum_fraction=0.05,
):
    """Increase bottle confidence when its bbox contains bottle HSV colour."""

    raw = float(raw_confidence)
    bonus = float(bonus)
    threshold = float(minimum_fraction)

    if not 0.0 <= raw <= 1.0:
        raise ValueError(
            "raw confidence must be between zero and one"
        )

    fraction = hsv_fraction_in_bbox(
        frame,
        bbox,
    )

    applied = (
        bonus
        if fraction >= threshold
        else 0.0
    )

    return (
        min(1.0, raw + applied),
        fraction,
        applied,
    )


def independent_hsv_bottle_candidates(
    frame,
    minimum_area=45.0,
    minimum_height=12,
    minimum_aspect_ratio=1.10,
):
    """Detect bottle-coloured regions directly from the entire image.

    This function is independent of YOLO.  It creates its own bounding
    boxes from connected HSV regions.
    """

    mask = bottle_hsv_mask(frame)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:

        contour_area = float(
            cv2.contourArea(contour)
        )

        if contour_area < float(minimum_area):
            continue

        x, y, width, height = cv2.boundingRect(
            contour
        )

        if height < int(minimum_height):
            continue

        if width <= 0:
            continue

        aspect_ratio = (
            float(height) /
            float(width)
        )

        # Bottle is vertically elongated.
        # This also rejects most small blue horizontal artefacts.
        if aspect_ratio < float(minimum_aspect_ratio):
            continue

        bbox_area = float(
            width * height
        )

        if bbox_area <= 0:
            continue

        fill_fraction = (
            contour_area /
            bbox_area
        )

        candidates.append(
            {
                "bbox": {
                    "x1": float(x),
                    "y1": float(y),
                    "x2": float(x + width),
                    "y2": float(y + height),
                },
                "area": contour_area,
                "fill_fraction": fill_fraction,
                "aspect_ratio": aspect_ratio,
            }
        )

    # Larger / more convincing bottle-colour regions first.
    candidates.sort(
        key=lambda item: (
            item["area"],
            item["fill_fraction"],
        ),
        reverse=True,
    )

    return candidates


def bbox_overlap_over_smaller(box_a, box_b):
    """Intersection area divided by the smaller bbox area.

    This is preferable to IoU here because the HSV body box is often
    narrower than the YOLO whole-object box.
    """

    ax1 = float(box_a["x1"])
    ay1 = float(box_a["y1"])
    ax2 = float(box_a["x2"])
    ay2 = float(box_a["y2"])

    bx1 = float(box_b["x1"])
    by1 = float(box_b["y1"])
    bx2 = float(box_b["x2"])
    by2 = float(box_b["y2"])

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(
        0.0,
        (ax2 - ax1) * (ay2 - ay1),
    )

    area_b = max(
        0.0,
        (bx2 - bx1) * (by2 - by1),
    )

    smaller = min(
        area_a,
        area_b,
    )

    if smaller <= 0.0:
        return 0.0

    return intersection / smaller
