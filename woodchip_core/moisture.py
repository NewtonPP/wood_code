# woodchip_core/moisture.py
"""
Moisture crop preprocessing + aggregation (pure numpy + cv2, no TRT/CUDA).

The class labels are passed in by the caller rather than read from a Jetson
file path, so the cloud inference service can supply them from the model bundle.
"""

import numpy as np
import cv2

from . import config


def prep_crop(crop_bgr, size, mean=None, std=None):
    mean = config.MEAN if mean is None else mean
    std = config.STD if std is None else std
    crop = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    crop = (crop - mean) / std
    return crop.astype(np.float32)  # HWC


def topk_crops(frame_bgr, boxes_xyxy, scores, topk,
               min_box_px=config.MOISTURE_MIN_BOX_PX,
               input_size=config.MOISTURE_INPUT_SIZE,
               mean=None, std=None):
    """
    Return:
      crops_bhwc (N,H,W,3) float32 normalized, or None
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
        if bw < min_box_px or bh < min_box_px:
            continue

        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue

        crops.append(prep_crop(crop_bgr, input_size, mean=mean, std=std))
        used.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": float(scores[idx])})

        if len(crops) >= topk:
            break

    if len(crops) == 0:
        return None, []
    return np.stack(crops, axis=0).astype(np.float32), used


def moisture_aggregate(probs, classes=None):
    mean_probs = np.mean(probs, axis=0)
    pred_idx = int(np.argmax(mean_probs))
    if classes and pred_idx < len(classes):
        pred_label = classes[pred_idx]
    else:
        pred_label = str(pred_idx)
    return mean_probs, pred_idx, pred_label
