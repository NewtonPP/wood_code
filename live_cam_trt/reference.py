# live_cam_trt/reference.py
"""
Reference-object (white disk) detection and online pixel/mm scale estimation.
Mutates calibration state in ``config`` (PIXELS_PER_MM, UNITS, SCALE_BUF).
"""

import numpy as np
import cv2

from . import config


def detect_white_reference_object(frame_bgr):
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 80, 40], np.uint8)
    upper = np.array([130, 225, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 2:
        contours, _ = cnts
    else:
        _, contours, _ = cnts

    best = None
    best_score = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        if area > 0.5 * h * w:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / float(bh + 1e-6)
        if aspect < 0.6 or aspect > 1.6:
            continue

        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        circ = 4.0 * np.pi * area / (perim * perim)

        score = circ * area
        if score > best_score:
            (cx_f, cy_f), radius = cv2.minEnclosingCircle(cnt)
            best = (int(cx_f), int(cy_f), float(2.0 * radius))
            best_score = score

    return best


def update_scale_from_reference(frame_bgr):
    ref = detect_white_reference_object(frame_bgr)
    if ref is None or config.REF_DIAM_MM <= 0:
        return None

    cx, cy, diameter_px = ref
    ppm_new = diameter_px / float(config.REF_DIAM_MM)

    config.SCALE_BUF.append(ppm_new)
    if len(config.SCALE_BUF) > config.SCALE_ROLLING_MAX:
        del config.SCALE_BUF[: len(config.SCALE_BUF) - config.SCALE_ROLLING_MAX]

    if config.PIXELS_PER_MM is None:
        config.PIXELS_PER_MM = ppm_new
    else:
        config.PIXELS_PER_MM = 0.9 * config.PIXELS_PER_MM + 0.1 * ppm_new

    config.UNITS = "mm"
    return cx, cy, diameter_px
