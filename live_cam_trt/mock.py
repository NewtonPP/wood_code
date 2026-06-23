# live_cam_trt/mock.py
"""
Synthetic ("mock") inference loop for local development without TensorRT/CUDA.

It publishes the same shared-state shapes as the real loop (LATEST_FRAME,
LATEST_STATS, HIST_DATA, LATEST_MOISTURE, LATEST_HEALTH) using random but
plausible data, so the FastAPI backend and the React frontend can be exercised
end-to-end on a dev machine (e.g. macOS).

Enable with WOODCHIP_FAKE_INFERENCE=1, or it is selected automatically when
tensorrt/pycuda are not installed.
"""

import os
import time
import numpy as np
import cv2

from . import config
from . import state
from .overlay import compute_diameter_stats, draw_stats_panel, draw_color_legend


def _now() -> float:
    return float(time.time())


def _mock_classes():
    # Prefer the real classes file if present; otherwise use sensible defaults.
    path = config.MOIST_CLASSES_PATH
    if path and os.path.exists(path):
        try:
            with open(path, "r") as f:
                cls = [l.strip() for l in f if l.strip()]
            if cls:
                return cls
        except OSError:
            pass
    return ["Dry", "Normal", "Wet"]


MOCK_CLASSES = _mock_classes()

# A fixed, calibrated scale so the UI shows mm + a populated histogram.
MOCK_PX_PER_MM = 4.0
FRAME_W, FRAME_H = 720, 540


def _draw_frame(boxes, d_vals, thr_mm, enabled):
    canvas = np.full((FRAME_H, FRAME_W, 3), 26, np.uint8)

    # reference disk (white circle)
    cv2.circle(canvas, (90, FRAME_H - 90), 36, (255, 255, 255), 2)
    cv2.circle(canvas, (90, FRAME_H - 90), 3, (0, 0, 255), -1)

    for (x1, y1, x2, y2), d in zip(boxes, d_vals):
        oversized = enabled and d > thr_mm
        color = (0, 0, 255) if oversized else (0, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    cv2.putText(canvas, "MOCK FEED", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return canvas


def run_mock_inference_loop(headless: bool = True):
    rng = np.random.default_rng()

    # Pretend we calibrated against the reference disk.
    config.PIXELS_PER_MM = MOCK_PX_PER_MM
    config.UNITS = "mm"

    D_buf = []
    start_ts = _now()
    fps = 10.0
    frame_idx = 0
    processed_idx = 0

    while True:
        frame_idx += 1
        processed_idx += 1

        thr_mm = float(config.ALARM_THRESHOLD_MM)
        enabled = bool(config.ALARM_ENABLED)

        # --- synthesize a handful of chip detections per frame ---
        n_chips = int(rng.integers(4, 10))
        d_vals = np.clip(rng.normal(45.0, 12.0, size=n_chips), 5.0, float(config.HIST_D_MAX_MM))
        boxes = []
        for d in d_vals:
            side = int(d * MOCK_PX_PER_MM / 1.4142)
            x1 = int(rng.integers(10, max(11, FRAME_W - side - 10)))
            y1 = int(rng.integers(10, max(11, FRAME_H - side - 10)))
            boxes.append((x1, y1, x1 + side, y1 + side))

        D_buf.extend(d_vals.tolist())
        if len(D_buf) > config.ROLLING_MAX:
            del D_buf[: len(D_buf) - config.ROLLING_MAX]

        # --- alarm ---
        max_d = float(np.max(d_vals)) if len(d_vals) else None
        alarm_active = bool(enabled and max_d is not None and max_d > thr_mm)
        config.ALARM_ACTIVE = alarm_active
        config.ALARM_MAX_D_MM = max_d

        # --- stats ---
        scale_mean = MOCK_PX_PER_MM
        scale_std = 0.2
        stats_dict = compute_diameter_stats(D_buf, config.UNITS, scale_mean=scale_mean, scale_std=scale_std)
        if stats_dict is not None:
            stats_dict["alarm_active"] = alarm_active
            stats_dict["alarm_threshold_mm"] = thr_mm
            stats_dict["alarm_max_d_mm"] = max_d
            stats_dict["ref_diam_mm"] = float(config.REF_DIAM_MM)

        # --- histogram (diameter, mm) ---
        hist_data = None
        if len(D_buf) > 10:
            vals = np.array(D_buf, dtype=np.float32)
            edges = np.linspace(0.0, float(config.HIST_D_MAX_MM), config.HIST_NUM_BINS + 1, dtype=np.float32)
            counts, _ = np.histogram(vals, bins=edges)
            centers = 0.5 * (edges[:-1] + edges[1:])
            pct_over = float(100.0 * np.mean(vals > thr_mm))
            hist_data = {
                "ready": True,
                "timestamp": _now(),
                "units": "mm",
                "mode": "diameter",
                "diameter": {"bins": centers.tolist(), "counts": counts.astype(int).tolist()},
                "threshold_x": thr_mm if enabled else None,
                "pct_over_threshold": pct_over if enabled else None,
                "alarm_threshold_mm": thr_mm,
                "alarm_enabled": enabled,
                "hist_num_bins": int(config.HIST_NUM_BINS),
                "hist_d_max_mm": float(config.HIST_D_MAX_MM),
            }

        # --- moisture ---
        moist_data = None
        if config.MOISTURE_ENABLED:
            logits = rng.normal(0.0, 1.0, size=len(MOCK_CLASSES))
            ex = np.exp(logits - np.max(logits))
            probs = ex / np.sum(ex)
            pred_idx = int(np.argmax(probs))
            moist_data = {
                "ready": True,
                "timestamp": _now(),
                "classes": MOCK_CLASSES,
                "mean_pred": MOCK_CLASSES[pred_idx],
                "mean_pred_index": pred_idx,
                "mean_probs": {MOCK_CLASSES[i]: float(probs[i]) for i in range(len(MOCK_CLASSES))},
                "topk": int(config.MOISTURE_TOPK),
                "boxes_used": min(n_chips, int(config.MOISTURE_TOPK)),
                "per_box": [],
            }

        # --- frame ---
        vis = _draw_frame(boxes, d_vals, thr_mm, enabled)
        if not headless:
            disp = draw_stats_panel(vis.copy(), D_buf, config.UNITS, scale_mean=scale_mean, scale_std=scale_std)
            disp = draw_color_legend(disp)
            vis = disp

        # --- health ---
        health_dict = {
            "ready": True,
            "timestamp": _now(),
            "uptime_sec": float(_now() - start_ts),
            "camera_ok": True,
            "cam_dev": "mock://camera",
            "last_frame_ts": _now(),
            "last_detr_ts": _now(),
            "last_moist_ts": _now() if moist_data else None,
            "last_hist_pub_ts": _now() if hist_data else None,
            "fps_smoothed": float(fps),
            "frame_idx": int(frame_idx),
            "processed_idx": int(processed_idx),
            "infer_every_n_frames": int(config.INFER_EVERY_N_FRAMES),
            "units": str(config.UNITS),
            "alarm_active": alarm_active,
            "alarm_threshold_mm": thr_mm,
            "alarm_max_d_mm": max_d,
            "last_error": None,
        }

        with state.STATE_LOCK:
            state.LATEST_FRAME = vis
            state.LATEST_STATS = stats_dict if stats_dict is not None else {"ready": False, "timestamp": _now()}
            if hist_data is not None:
                state.HIST_DATA = hist_data
            if moist_data is not None:
                state.LATEST_MOISTURE = moist_data
            state.LATEST_HEALTH = health_dict

        time.sleep(0.2)
