# live_cam_trt/config.py
"""
All tunable configuration constants + runtime config get/update.

These values are intentionally module-level so they can be mutated at runtime
(via update_runtime_config / calibration) and read live by the inference loop.
Other modules MUST reference them qualified (e.g. ``config.CONF_THR``) so they
observe runtime changes.
"""

import os
import numpy as np


# =================== CONFIG ===================
ENGINE_PATH = "/home/huser/Amir/detr-deploy/models/detr_resnet101_fp16.engine"
MOIST_ENGINE_PATH = "/home/huser/Amir/detr-deploy/models/moisture/moistnetlite_fp16.engine"
MOIST_CLASSES_PATH = "/home/huser/Amir/detr-deploy/models/moisture/moistnetlite_classes.txt"

OUT_DIR = "/home/huser/Amir/detr-deploy/live_outputs"
# Best-effort: the Jetson output dir won't exist on a dev machine (e.g. macOS).
# Guard so the package stays importable everywhere (the backend imports it for
# config/state access without running the loop).
try:
    os.makedirs(OUT_DIR, exist_ok=True)
except OSError:
    pass

CAM_DEV = "/dev/video0"
CAM_W, CAM_H, CAM_FPS = 1480, 900, 10  # UPDATED: Camera FPS = 20

# UPDATED: processed FPS control (camera 20 FPS, infer every 2 frames => ~10 FPS)
INFER_EVERY_N_FRAMES = 2

# DETR input handling (your old settings)
ENGINE_INPUT_MODE = "fixed"
FIXED_H, FIXED_W = 800, 1333
SHORTEST_EDGE, LONGEST_EDGE = 800, 1333

CONF_THR, NMS_IOU = 0.5, 0.5

# --- Reference object config ---
REF_DIAM_MM = 110.0   # real diameter of your blue disk in millimeters (user-configurable)
PIXELS_PER_MM = None  # estimated online from the blue reference object
UNITS = "pixels"

# Rolling buffer of px/mm estimates
SCALE_BUF = []
SCALE_ROLLING_MAX = 500

# Alarm
ALARM_THRESHOLD_MM = 60.0
ALARM_ENABLED = True
ALARM_ACTIVE = False
ALARM_MAX_D_MM = None

# Histogram mode (kept for API compatibility; will publish only diameter histogram)
HISTOGRAM_MODE = "diameter"  # "length" | "width" | "diameter"

# UPDATED: rolling sample window
ROLLING_MAX = 2000

# UPDATED: histogram update cadence
HIST_UPDATE_INTERVAL = 1.0  # 1 Hz

# UPDATED: histogram bins and fixed range (in mm once calibrated)
HIST_NUM_BINS = 20
HIST_D_MAX_MM = 120.0  # choose based on your spec (e.g., 80–120 mm); used when UNITS == "mm"

# UPDATED: optional EMA smoothing for histogram stability
HIST_EMA_ENABLED = True
HIST_EMA_ALPHA = 0.2

# Moisture runtime config (unchanged except its cadence is tied to processed frames below)
MOISTURE_ENABLED = True
MOISTURE_TOPK = 8
MOISTURE_EVERY_N_FRAMES = 3
MOISTURE_INPUT_SIZE = 224
MOISTURE_MIN_BOX_PX = 12
MOISTURE_FALLBACK_FULLFRAME = True  # if no boxes, still run moisture on full frame

# Normalization (ImageNet)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# =================== WEB CONFIG HELPERS ===================
def get_runtime_config():
    return {
        "conf_thr": float(CONF_THR),
        "nms_iou": float(NMS_IOU),
        "alarm_threshold_mm": float(ALARM_THRESHOLD_MM),
        "alarm_enabled": bool(ALARM_ENABLED),
        "ref_diam_mm": float(REF_DIAM_MM),
        "histogram_mode": HISTOGRAM_MODE,

        "moisture_enabled": bool(MOISTURE_ENABLED),
        "moisture_topk": int(MOISTURE_TOPK),
        "moisture_every_n_frames": int(MOISTURE_EVERY_N_FRAMES),

        # expose agreed knobs (safe extras; clients can ignore)
        "camera_fps": int(CAM_FPS),
        "infer_every_n_frames": int(INFER_EVERY_N_FRAMES),
        "rolling_max": int(ROLLING_MAX),
        "hist_update_interval": float(HIST_UPDATE_INTERVAL),
        "hist_num_bins": int(HIST_NUM_BINS),
        "hist_d_max_mm": float(HIST_D_MAX_MM),
        "hist_ema_enabled": bool(HIST_EMA_ENABLED),
        "hist_ema_alpha": float(HIST_EMA_ALPHA),
    }


def update_runtime_config(
    conf_thr=None,
    nms_iou=None,
    alarm_threshold_mm=None,
    alarm_enabled=None,
    ref_diam_mm=None,
    histogram_mode=None,
    moisture_enabled=None,
    moisture_topk=None,
    moisture_every_n_frames=None,
    **_ignored
):
    global CONF_THR, NMS_IOU, ALARM_THRESHOLD_MM, ALARM_ENABLED, REF_DIAM_MM, HISTOGRAM_MODE
    global MOISTURE_ENABLED, MOISTURE_TOPK, MOISTURE_EVERY_N_FRAMES

    if conf_thr is not None:
        try:
            CONF_THR = max(0.0, min(1.0, float(conf_thr)))
        except (ValueError, TypeError):
            pass

    if nms_iou is not None:
        try:
            NMS_IOU = max(0.0, min(1.0, float(nms_iou)))
        except (ValueError, TypeError):
            pass

    if alarm_threshold_mm is not None:
        try:
            ALARM_THRESHOLD_MM = max(0.0, float(alarm_threshold_mm))
        except (ValueError, TypeError):
            pass

    if alarm_enabled is not None:
        ALARM_ENABLED = bool(alarm_enabled)

    if ref_diam_mm is not None:
        try:
            v = float(ref_diam_mm)
            if v > 0:
                REF_DIAM_MM = v
        except (ValueError, TypeError):
            pass

    # kept for compatibility; UI can still set it, but we publish only diameter histogram anyway
    if histogram_mode is not None:
        hm = str(histogram_mode).lower()
        if hm in ("length", "width", "diameter"):
            HISTOGRAM_MODE = hm

    if moisture_enabled is not None:
        MOISTURE_ENABLED = bool(moisture_enabled)
    if moisture_topk is not None:
        try:
            MOISTURE_TOPK = max(1, int(moisture_topk))
        except Exception:
            pass
    if moisture_every_n_frames is not None:
        try:
            MOISTURE_EVERY_N_FRAMES = max(1, int(moisture_every_n_frames))
        except Exception:
            pass
