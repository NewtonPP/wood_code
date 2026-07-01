# woodchip_core/processor.py
"""
``FrameProcessor`` — the platform-independent post-processing pipeline.

Given the raw model outputs for a single frame (DETR boxes+scores, and a
moisture-inference callable), it reproduces exactly what the Jetson inference
loop used to compute inline: NMS, sizing, online px/mm calibration from the
reference disk, rolling diameter statistics, a throttled + EMA-smoothed
histogram, the oversize alarm, and moisture aggregation.

All *per-stream* state (calibration, rolling buffers, histogram EMA, alarm
latch, counters) lives on the instance, so the cloud inference service can keep
one ``FrameProcessor`` per camera/device and they never cross-contaminate.

The returned dict carries the same payload shapes the React frontend already
consumes (``stats`` == LATEST_STATS, ``histogram`` == HIST_DATA, ``moisture``
== LATEST_MOISTURE) plus a compact ``boxes`` list for the browser to draw the
overlay locally.
"""

import time

import numpy as np

from . import config as default_config
from .geometry import nms_np, compute_LWD
from .reference import detect_white_reference_object
from .moisture import topk_crops, prep_crop, moisture_aggregate
from .overlay import compute_diameter_stats


def _now():
    return float(time.time())


class FrameProcessor:
    def __init__(self, cfg=default_config, moisture_classes=None, initial_pixels_per_mm=None):
        # ``cfg`` is the tunable-rules module/object (defaults to woodchip_core.config).
        self.cfg = cfg
        self.moisture_classes = moisture_classes

        # --- per-stream calibration ---
        self.pixels_per_mm = initial_pixels_per_mm
        self.units = "mm" if initial_pixels_per_mm else "pixels"
        self.scale_buf = []

        # --- rolling diameter buffer ---
        self.D_buf = []

        # --- alarm latch ---
        self.alarm_active = False
        self.alarm_max_d_mm = None

        # --- histogram throttle + EMA ---
        self._last_hist_time = 0.0
        self._prev_hist_counts = None
        self._prev_hist_bins = None
        self._last_hist = None

        # --- moisture cadence ---
        self._last_moisture = None

        # --- telemetry counters ---
        self._start_ts = _now()
        self._frame_idx = 0
        self._processed_idx = 0
        self._fps = 0.0
        self._t_prev = time.time()
        self._last_detr_ts = None
        self._last_moist_ts = None
        self._last_hist_pub_ts = None

    # ------------------------------------------------------------------ #
    def set_scale(self, pixels_per_mm):
        """Force a calibration scale (used by the mock backend for a mm demo)."""
        self.pixels_per_mm = float(pixels_per_mm)
        self.units = "mm"

    # ------------------------------------------------------------------ #
    def process(self, frame_bgr, boxes_xyxy, scores, moisture_infer=None):
        cfg = self.cfg
        self._frame_idx += 1
        self._processed_idx += 1

        # --- online calibration from the white reference disk ---
        ref_out = self._update_calibration(frame_bgr)
        scale_mean = float(np.mean(self.scale_buf)) if self.scale_buf else None
        scale_std = float(np.std(self.scale_buf)) if self.scale_buf else None

        # --- NMS ---
        boxes_xyxy = np.asarray(boxes_xyxy, np.float32).reshape(-1, 4)
        scores = np.asarray(scores, np.float32).reshape(-1)
        if boxes_xyxy.shape[0] > 0:
            keep = nms_np(boxes_xyxy, scores, iou_thr=cfg.NMS_IOU)
            boxes_xyxy = boxes_xyxy[keep]
            scores = scores[keep]
            self._last_detr_ts = _now()

        # --- sizing ---
        L, W, D = compute_LWD(boxes_xyxy, self.pixels_per_mm)
        if D.size:
            self.D_buf.extend(D.tolist())
            if len(self.D_buf) > cfg.ROLLING_MAX:
                del self.D_buf[: len(self.D_buf) - cfg.ROLLING_MAX]

        # --- oversize alarm (mm, after calibration) ---
        alarm_active = False
        max_d_mm = None
        if cfg.ALARM_ENABLED and self.pixels_per_mm and D.size > 0:
            max_d_mm = float(np.max(D))
            if max_d_mm > cfg.ALARM_THRESHOLD_MM:
                alarm_active = True
        self.alarm_active = alarm_active
        self.alarm_max_d_mm = max_d_mm

        # --- diameter statistics ---
        stats = compute_diameter_stats(self.D_buf, self.units, scale_mean=scale_mean, scale_std=scale_std)
        if stats is not None:
            stats["alarm_active"] = bool(alarm_active)
            stats["alarm_threshold_mm"] = float(cfg.ALARM_THRESHOLD_MM)
            stats["alarm_max_d_mm"] = float(max_d_mm) if max_d_mm is not None else None
            stats["ref_diam_mm"] = float(cfg.REF_DIAM_MM)
        else:
            stats = {"ready": False, "timestamp": _now(), "error": "No detections yet."}

        # --- histogram (throttled to HIST_UPDATE_INTERVAL, EMA-smoothed) ---
        histogram = self._maybe_build_histogram()

        # --- moisture (cadence tied to processed frames) ---
        moisture = self._maybe_run_moisture(frame_bgr, boxes_xyxy, scores, moisture_infer)

        # --- FPS smoothing ---
        now = time.time()
        fps_inst = 1.0 / max(1e-3, now - self._t_prev)
        self._fps = 0.9 * self._fps + 0.1 * fps_inst
        self._t_prev = now

        # --- compact boxes for the browser overlay ---
        box_list = []
        for (x1, y1, x2, y2), sc, d_val in zip(boxes_xyxy.tolist(), scores.tolist(), D.tolist()):
            oversized = bool(
                cfg.ALARM_ENABLED and self.pixels_per_mm is not None and d_val > cfg.ALARM_THRESHOLD_MM
            )
            box_list.append({
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "score": float(sc), "diameter": float(d_val), "oversized": oversized,
            })

        return {
            "ready": True,
            "timestamp": _now(),
            "units": self.units,
            "stats": stats,
            "histogram": histogram,
            "moisture": moisture,
            "health": self._health(),
            "boxes": box_list,
            "reference": ref_out,
            "frame_size": {"w": int(frame_bgr.shape[1]), "h": int(frame_bgr.shape[0])},
        }

    # ------------------------------------------------------------------ #
    def _update_calibration(self, frame_bgr):
        cfg = self.cfg
        ref = detect_white_reference_object(frame_bgr)
        if ref is None or cfg.REF_DIAM_MM <= 0:
            return None
        cx, cy, diameter_px = ref
        ppm_new = diameter_px / float(cfg.REF_DIAM_MM)

        self.scale_buf.append(ppm_new)
        if len(self.scale_buf) > cfg.SCALE_ROLLING_MAX:
            del self.scale_buf[: len(self.scale_buf) - cfg.SCALE_ROLLING_MAX]

        if self.pixels_per_mm is None:
            self.pixels_per_mm = ppm_new
        else:
            self.pixels_per_mm = 0.9 * self.pixels_per_mm + 0.1 * ppm_new
        self.units = "mm"
        return {"cx": int(cx), "cy": int(cy), "diameter_px": float(diameter_px)}

    # ------------------------------------------------------------------ #
    def _maybe_build_histogram(self):
        cfg = self.cfg
        now = time.time()
        if (now - self._last_hist_time) < float(cfg.HIST_UPDATE_INTERVAL) or len(self.D_buf) <= 10:
            return self._last_hist  # keep last so the UI never blanks between updates
        self._last_hist_time = now

        vals = np.array(self.D_buf, dtype=np.float32)
        if self.units == "mm":
            upper = max(1.0, float(cfg.HIST_D_MAX_MM))
        else:
            upper = float(np.percentile(vals, 99.0))
            if upper <= 0:
                upper = float(np.max(vals)) if len(vals) > 0 else 1.0
            upper = max(1.0, upper)
        edges = np.linspace(0.0, upper, cfg.HIST_NUM_BINS + 1, dtype=np.float32)
        hist, _ = np.histogram(vals, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])

        if cfg.HIST_EMA_ENABLED:
            if (self._prev_hist_counts is not None and self._prev_hist_bins is not None
                    and len(self._prev_hist_counts) == len(hist)
                    and np.allclose(self._prev_hist_bins, centers)):
                sm = (1.0 - cfg.HIST_EMA_ALPHA) * self._prev_hist_counts + cfg.HIST_EMA_ALPHA * hist.astype(np.float32)
                hist_out = np.rint(sm).astype(int)
            else:
                hist_out = hist.astype(int)
            self._prev_hist_counts = hist_out.astype(np.float32)
            self._prev_hist_bins = centers.copy()
        else:
            hist_out = hist.astype(int)

        threshold_x = None
        pct_over_threshold = None
        if cfg.ALARM_ENABLED and self.units == "mm" and len(self.D_buf) > 0:
            thr = float(cfg.ALARM_THRESHOLD_MM)
            threshold_x = thr
            pct_over_threshold = float(100.0 * np.mean(vals > thr))

        self._last_hist_pub_ts = _now()
        self._last_hist = {
            "ready": True,
            "timestamp": _now(),
            "units": self.units,
            "mode": "diameter",
            "diameter": {"bins": centers.tolist(), "counts": hist_out.tolist()},
            "threshold_x": threshold_x,
            "pct_over_threshold": pct_over_threshold,
            "alarm_threshold_mm": float(cfg.ALARM_THRESHOLD_MM),
            "alarm_enabled": bool(cfg.ALARM_ENABLED),
            "hist_update_interval": float(cfg.HIST_UPDATE_INTERVAL),
            "hist_num_bins": int(cfg.HIST_NUM_BINS),
            "hist_d_max_mm": float(cfg.HIST_D_MAX_MM) if self.units == "mm" else None,
            "hist_ema_enabled": bool(cfg.HIST_EMA_ENABLED),
            "hist_ema_alpha": float(cfg.HIST_EMA_ALPHA) if cfg.HIST_EMA_ENABLED else None,
        }
        return self._last_hist

    # ------------------------------------------------------------------ #
    def _maybe_run_moisture(self, frame_bgr, boxes_xyxy, scores, moisture_infer):
        cfg = self.cfg
        if not cfg.MOISTURE_ENABLED or moisture_infer is None:
            return self._last_moisture
        if self._processed_idx % max(1, cfg.MOISTURE_EVERY_N_FRAMES) != 0:
            return self._last_moisture

        crops_bhwc, used_boxes = topk_crops(
            frame_bgr, boxes_xyxy, scores, cfg.MOISTURE_TOPK,
            min_box_px=cfg.MOISTURE_MIN_BOX_PX, input_size=cfg.MOISTURE_INPUT_SIZE,
        )
        if (crops_bhwc is None or len(used_boxes) == 0) and cfg.MOISTURE_FALLBACK_FULLFRAME:
            crops_bhwc = np.stack([prep_crop(frame_bgr, cfg.MOISTURE_INPUT_SIZE)], axis=0).astype(np.float32)
            used_boxes = []

        if crops_bhwc is None:
            self._last_moisture = {
                "ready": False, "timestamp": _now(),
                "error": "No crops available for moisture (no boxes and fallback disabled).",
                "topk": int(cfg.MOISTURE_TOPK), "boxes_used": 0,
            }
            return self._last_moisture

        probs = moisture_infer(crops_bhwc)
        if probs is None:
            return self._last_moisture
        probs = np.asarray(probs, np.float32)

        mean_probs, pred_idx, pred_label = moisture_aggregate(probs, self.moisture_classes)
        self._last_moist_ts = _now()
        classes = self.moisture_classes

        per_box = []
        if used_boxes:
            for i, ub in enumerate(used_boxes):
                pi = int(np.argmax(probs[i]))
                pl = classes[pi] if classes and pi < len(classes) else str(pi)
                per_box.append({
                    "box": ub,
                    "pred": pl,
                    "pred_index": pi,
                    "probs": (
                        {classes[j]: float(probs[i][j]) for j in range(probs.shape[1])}
                        if classes and len(classes) == probs.shape[1]
                        else [float(x) for x in probs[i]]
                    ),
                })

        self._last_moisture = {
            "ready": True,
            "timestamp": _now(),
            "classes": classes if classes else None,
            "mean_pred": pred_label,
            "mean_pred_index": pred_idx,
            "mean_probs": (
                {classes[i]: float(mean_probs[i]) for i in range(len(mean_probs))}
                if classes and len(classes) == len(mean_probs)
                else [float(x) for x in mean_probs]
            ),
            "topk": int(cfg.MOISTURE_TOPK),
            "boxes_used": int(len(used_boxes)),
            "per_box": per_box,
        }
        return self._last_moisture

    # ------------------------------------------------------------------ #
    def _health(self):
        cfg = self.cfg
        return {
            "ready": True,
            "timestamp": _now(),
            "uptime_sec": float(_now() - self._start_ts),
            "camera_ok": True,
            "cam_dev": "browser://getUserMedia",
            "last_frame_ts": _now(),
            "last_detr_ts": self._last_detr_ts,
            "last_moist_ts": self._last_moist_ts,
            "last_hist_pub_ts": self._last_hist_pub_ts,
            "fps_smoothed": float(self._fps),
            "frame_idx": int(self._frame_idx),
            "processed_idx": int(self._processed_idx),
            "infer_every_n_frames": int(cfg.INFER_EVERY_N_FRAMES),
            "units": str(self.units),
            "alarm_active": bool(self.alarm_active),
            "alarm_threshold_mm": float(cfg.ALARM_THRESHOLD_MM),
            "alarm_max_d_mm": float(self.alarm_max_d_mm) if self.alarm_max_d_mm is not None else None,
            "last_error": None,
        }
