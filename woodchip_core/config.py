# woodchip_core/config.py
"""
Platform-independent (CUDA-free) runtime configuration for the wood-chip
post-processing pipeline.

This module holds *only* the tunable knobs + model-input constants that the
post-processing needs. Unlike ``live_cam_trt/config.py`` it intentionally
contains **no** Jetson/TensorRT specifics (engine paths, camera device, output
dirs) so it can be imported anywhere — the cloud inference service, a dev Mac,
or the Jetson device.

Per-*stream* state (pixel/mm calibration, rolling diameter buffers, alarm
latch, histogram EMA) does NOT live here — it lives on a
``woodchip_core.processor.FrameProcessor`` instance so multiple camera streams
stay isolated from each other. Only globally-shared, operator-tunable rules
(confidence/NMS thresholds, alarm threshold, moisture cadence, …) are module
globals, mutated via :func:`update_runtime_config`.
"""

import numpy as np


# =================== Model-input constants (DETR) ===================
# These mirror the values the model was exported with; they are not operator
# tunable. Kept here so the ONNX backend and the device share one definition.
FIXED_H, FIXED_W = 800, 1333
SHORTEST_EDGE, LONGEST_EDGE = 800, 1333

# Normalization (ImageNet)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# =================== Operator-tunable runtime knobs ===================
CONF_THR, NMS_IOU = 0.5, 0.5

# --- Reference object / calibration ---
REF_DIAM_MM = 110.0   # real diameter of the blue calibration disk (mm)
SCALE_ROLLING_MAX = 500

# --- Alarm ---
ALARM_THRESHOLD_MM = 60.0
ALARM_ENABLED = True

# --- Histogram ---
HISTOGRAM_MODE = "diameter"  # kept for API compatibility; only diameter is published
ROLLING_MAX = 2000
HIST_UPDATE_INTERVAL = 1.0   # seconds (1 Hz)
HIST_NUM_BINS = 20
HIST_D_MAX_MM = 120.0        # fixed upper range once calibrated to mm
HIST_EMA_ENABLED = True
HIST_EMA_ALPHA = 0.2

# --- Moisture ---
MOISTURE_ENABLED = True
MOISTURE_TOPK = 8
MOISTURE_EVERY_N_FRAMES = 3
MOISTURE_INPUT_SIZE = 224
MOISTURE_MIN_BOX_PX = 12
MOISTURE_FALLBACK_FULLFRAME = True

# Informational only (the cloud does not skip frames — the browser throttles
# capture FPS — but the value is exposed so the rules UI stays consistent).
INFER_EVERY_N_FRAMES = 1


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

    if histogram_mode is not None:
        hm = str(histogram_mode).lower()
        if hm in ("length", "width", "diameter"):
            HISTOGRAM_MODE = hm

    if moisture_enabled is not None:
        MOISTURE_ENABLED = bool(moisture_enabled)
    if moisture_topk is not None:
        try:
            MOISTURE_TOPK = max(1, int(moisture_topk))
        except (ValueError, TypeError):
            pass
    if moisture_every_n_frames is not None:
        try:
            MOISTURE_EVERY_N_FRAMES = max(1, int(moisture_every_n_frames))
        except (ValueError, TypeError):
            pass
