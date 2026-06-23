# live_cam_trt/geometry.py
"""
Math / preprocessing / postprocessing helpers (no TRT, no CUDA).
"""

import time
import numpy as np
import cv2

from . import config


def _now():
    return float(time.time())


def softmax_lastdim(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=-1, keepdims=True) + 1e-12)


def safe_softmax_2d(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)


def nms_np(boxes_xyxy, scores, iou_thr=0.5):
    if boxes_xyxy.shape[0] == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter + 1e-12
        iou = inter / union
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def resize_keep_aspect(h0, w0, shortest=config.SHORTEST_EDGE, longest=config.LONGEST_EDGE):
    s = shortest / float(min(h0, w0))
    if max(h0, w0) * s > longest:
        s = longest / float(max(h0, w0))
    return s, int(round(h0 * s)), int(round(w0 * s))


def preprocess_fixed(img_bgr):
    h0, w0 = img_bgr.shape[:2]
    s, Hr, Wr = resize_keep_aspect(h0, w0)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_resized = cv2.resize(img_rgb, (Wr, Hr), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((config.FIXED_H, config.FIXED_W, 3), np.float32)
    canvas[:Hr, :Wr] = img_resized
    canvas = (canvas - config.MEAN) / config.STD
    x = np.transpose(canvas, (2, 0, 1))[None, ...].astype(np.float32)
    return x, (h0, w0, Hr, Wr)


def postprocess_from_fixed(pred_boxes, logits, meta, conf_thr=0.5):
    h0, w0, Hr, Wr = meta

    logits = logits.reshape(1, -1, logits.shape[-1])
    probs = softmax_lastdim(logits)[0]

    scores = probs[:, :-1].max(axis=1)
    keep = np.where(scores >= conf_thr)[0]
    if keep.size == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

    boxes = pred_boxes[0][keep]
    scores = scores[keep]

    cx = boxes[:, 0] * config.FIXED_W
    cy = boxes[:, 1] * config.FIXED_H
    ww = boxes[:, 2] * config.FIXED_W
    hh = boxes[:, 3] * config.FIXED_H

    x1 = cx - ww / 2.0
    y1 = cy - hh / 2.0
    x2 = cx + ww / 2.0
    y2 = cy + hh / 2.0

    x1 = np.clip(x1, 0, Wr - 1)
    x2 = np.clip(x2, 0, Wr - 1)
    y1 = np.clip(y1, 0, Hr - 1)
    y2 = np.clip(y2, 0, Hr - 1)

    sx = w0 / float(Wr)
    sy = h0 / float(Hr)

    x1 *= sx
    x2 *= sx
    y1 *= sy
    y2 *= sy

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    boxes_xyxy[:, 0::2] = np.clip(boxes_xyxy[:, 0::2], 0, w0 - 1)
    boxes_xyxy[:, 1::2] = np.clip(boxes_xyxy[:, 1::2], 0, h0 - 1)

    return boxes_xyxy.astype(np.float32), scores.astype(np.float32)


def compute_LWD(boxes_xyxy):
    """Return L,W,D in pixels or mm depending on PIXELS_PER_MM."""
    if boxes_xyxy.shape[0] == 0:
        return (np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))
    arr = np.asarray(boxes_xyxy, np.float32)
    L = np.abs(arr[:, 2] - arr[:, 0])
    W = np.abs(arr[:, 3] - arr[:, 1])
    D = np.sqrt(L * L + W * W)
    if config.PIXELS_PER_MM:
        s = float(config.PIXELS_PER_MM)
        L, W, D = L / s, W / s, D / s
    return L, W, D
