#!/usr/bin/env python3
"""Color evidence used to reinforce low-confidence bottle detections."""

from __future__ import annotations

import numpy as np


def light_blue_fraction_bgr(frame, bbox):
    """Return the fraction of light-blue pixels inside a clipped BGR bbox."""
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("frame must be an HxWx3 BGR image")
    height, width = image.shape[:2]
    x1 = max(0, min(width, int(round(float(bbox["x1"])))))
    y1 = max(0, min(height, int(round(float(bbox["y1"])))))
    x2 = max(0, min(width, int(round(float(bbox["x2"])))))
    y2 = max(0, min(height, int(round(float(bbox["y2"])))))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = image[y1:y2, x1:x2, :3].astype(np.float32)
    blue = roi[:, :, 0]
    green = roi[:, :, 1]
    red = roi[:, :, 2]
    # The simulated bottle is a pale cyan-blue.  Requiring all channels to
    # remain fairly bright rejects dark navy objects, while the channel gaps
    # reject the grey floor and white highlights.
    mask = (
        (blue >= 135.0)
        & (green >= 100.0)
        & (red >= 55.0)
        & (blue >= green + 14.0)
        & (green >= red + 12.0)
    )
    return float(mask.mean())


def adjusted_bottle_confidence(
    frame,
    bbox,
    raw_confidence,
    bonus=0.5,
    minimum_fraction=0.18,
):
    """Apply a bounded bonus only when the detection ROI is visibly pale blue."""
    raw = float(raw_confidence)
    color_bonus = float(bonus)
    threshold = float(minimum_fraction)
    if not 0.0 <= raw <= 1.0:
        raise ValueError("raw confidence must be between zero and one")
    if color_bonus < 0.0 or not 0.0 <= threshold <= 1.0:
        raise ValueError("invalid color confidence parameters")
    fraction = light_blue_fraction_bgr(frame, bbox)
    applied = color_bonus if fraction >= threshold else 0.0
    return min(1.0, raw + applied), fraction, applied
