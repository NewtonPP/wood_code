# woodchip_core/reference.py
"""
Reference-object (blue disk) detection for online pixel/mm scale estimation.

Pure and side-effect free: it only *detects* the disk. The actual scale update
(EMA of px/mm) is owned by the caller — ``FrameProcessor`` for the cloud (per
stream), or ``live_cam_trt.reference.update_scale_from_reference`` for the
device (module-global, single camera).
"""

import numpy as np
import cv2


def detect_reference_object(frame_bgr):
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
