# live_cam_trt/moisture.py
"""
Moisture preprocessing / postprocessing helpers + class labels.
"""

import os
import numpy as np
import cv2

from . import config


def _read_moist_classes(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]


MOISTURE_CLASSES = _read_moist_classes(config.MOIST_CLASSES_PATH)


def _prep_moist_crop_bgr(crop_bgr: np.ndarray, size: int):
    crop = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    crop = (crop - config.MEAN) / config.STD
    return crop.astype(np.float32)  # HWC


def _boxes_to_topk_crops(frame_bgr: np.ndarray, boxes_xyxy: np.ndarray, scores: np.ndarray, topk: int):
    """
    Return:
      crops_bhwc (N,H,W,3) float32 normalized
      used_boxes list of dicts (coords + score)
    """
    h, w = frame_bgr.shape[:2]
    if boxes_xyxy is None or boxes_xyxy.shape[0] == 0:
        return None, []

    order = np.argsort(scores)[::-1]
    crops = []
    used = []

    for idx in order[: max(1, topk)]:
        x1, y1, x2, y2 = boxes_xyxy[idx].astype(int).tolist()
        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, 0, h - 1))

        bw = x2 - x1
        bh = y2 - y1
        if bw < config.MOISTURE_MIN_BOX_PX or bh < config.MOISTURE_MIN_BOX_PX:
            continue

        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue

        crops.append(_prep_moist_crop_bgr(crop_bgr, config.MOISTURE_INPUT_SIZE))
        used.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": float(scores[idx])})

        if len(crops) >= topk:
            break

    if len(crops) == 0:
        return None, []
    return np.stack(crops, axis=0).astype(np.float32), used


def _moisture_aggregate(probs: np.ndarray):
    mean_probs = np.mean(probs, axis=0)
    pred_idx = int(np.argmax(mean_probs))
    if MOISTURE_CLASSES and pred_idx < len(MOISTURE_CLASSES):
        pred_label = MOISTURE_CLASSES[pred_idx]
    else:
        pred_label = str(pred_idx)
    return mean_probs, pred_idx, pred_label
