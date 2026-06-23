# live_cam_trt/loop.py
"""
Unified inference loop: DETR (size + histogram + calibration + alarm) +
Moisture (TRT) in ONE thread/CUDA context.

CUDA/TRT stability is preserved exactly:
  * single global TRT logger (see engines.py)
  * buffers allocated per FULL binding shape after set_binding_shape
  * destroy TRT objects BEFORE ctx.pop()
  * one CUDA context per inference thread
"""

import time
import numpy as np
import cv2
import pycuda.driver as cuda
import traceback

from . import config
from . import state
from .engines import DETRTRT, MoistureTRT
from .geometry import (
    _now,
    preprocess_fixed,
    postprocess_from_fixed,
    nms_np,
    compute_LWD,
)
from .reference import update_scale_from_reference
from . import moisture as moisture_mod
from .moisture import _boxes_to_topk_crops, _prep_moist_crop_bgr, _moisture_aggregate
from .overlay import compute_diameter_stats, draw_stats_panel, draw_color_legend


# =================== MAIN LOOP ===================
def run_inference_loop(headless: bool = True):
    print("[live_cam_trt] Initializing CUDA in inference thread...", flush=True)
    cuda.init()
    dev = cuda.Device(0)
    ctx = dev.make_context()

    cap = None
    detr = None
    moist = None

    # --- Health/telemetry state (for /api/devices/status) ---
    start_ts = _now()
    last_frame_ts = None
    last_detr_ts = None
    last_moist_ts = None
    last_hist_pub_ts = None
    last_error = None

    try:
        print("[live_cam_trt] Opening camera:", config.CAM_DEV, flush=True)
        cap = cv2.VideoCapture(config.CAM_DEV, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_H)
        cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)

        for _ in range(5):
            cap.read()

        if not cap.isOpened():
            print("[live_cam_trt] ERROR: camera failed to open.", flush=True)
            err = f"camera failed to open: {config.CAM_DEV}"
            last_error = err

            with state.STATE_LOCK:
                state.LATEST_STATS = {"ready": False, "error": err, "timestamp": _now()}
                state.HIST_DATA = {"ready": False, "error": "camera failed to open", "timestamp": _now()}
                state.LATEST_MOISTURE = {"ready": False, "error": "camera failed to open", "timestamp": _now()}
                state.LATEST_HEALTH = {
                    "ready": False,
                    "timestamp": _now(),
                    "uptime_sec": float(_now() - start_ts),
                    "camera_ok": False,
                    "cam_dev": str(config.CAM_DEV),
                    "last_frame_ts": None,
                    "last_detr_ts": None,
                    "last_moist_ts": None,
                    "last_hist_pub_ts": None,
                    "fps_smoothed": 0.0,
                    "frame_idx": 0,
                    "processed_idx": 0,
                    "infer_every_n_frames": int(config.INFER_EVERY_N_FRAMES),
                    "units": str(config.UNITS),
                    "alarm_active": bool(config.ALARM_ACTIVE),
                    "alarm_max_d_mm": None,
                    "last_error": err,
                }
            return

        print("[live_cam_trt] Loading DETR TensorRT engine:", config.ENGINE_PATH, flush=True)
        detr = DETRTRT(config.ENGINE_PATH)
        print("[live_cam_trt] DETR engine loaded OK", flush=True)

        if config.MOISTURE_ENABLED:
            print("[live_cam_trt] Loading Moisture TensorRT engine:", config.MOIST_ENGINE_PATH, flush=True)
            moist = MoistureTRT(config.MOIST_ENGINE_PATH)
            print("[live_cam_trt] Moisture engine loaded OK", flush=True)

        L_buf, W_buf, D_buf = [], [], []
        fps, t_prev = 0.0, time.time()

        frame_idx = 0
        processed_idx = 0

        # Keep last inference results to overlay on skipped frames
        last_boxes_xyxy = np.zeros((0, 4), np.float32)
        last_scores = np.zeros((0,), np.float32)
        last_D = np.zeros((0,), np.float32)

        # Histogram throttling + EMA state (diameter only)
        last_hist_time = 0.0
        prev_hist_counts = None  # np.float32 array
        prev_hist_bins = None    # centers

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # keep loop alive, but publish health that we are not receiving frames
                last_error = "cap.read() failed"
                print("[live_cam_trt] WARNING: cap.read() failed, continuing...", flush=True)
                time.sleep(0.05)

                with state.STATE_LOCK:
                    state.LATEST_HEALTH = {
                        "ready": True,
                        "timestamp": _now(),
                        "uptime_sec": float(_now() - start_ts),
                        "camera_ok": True,  # camera object exists; frame read failed transiently
                        "cam_dev": str(config.CAM_DEV),
                        "last_frame_ts": last_frame_ts,
                        "last_detr_ts": last_detr_ts,
                        "last_moist_ts": last_moist_ts,
                        "last_hist_pub_ts": last_hist_pub_ts,
                        "fps_smoothed": float(fps),
                        "frame_idx": int(frame_idx),
                        "processed_idx": int(processed_idx),
                        "infer_every_n_frames": int(config.INFER_EVERY_N_FRAMES),
                        "units": str(config.UNITS),
                        "alarm_active": bool(config.ALARM_ACTIVE),
                        "alarm_max_d_mm": float(config.ALARM_MAX_D_MM) if config.ALARM_MAX_D_MM is not None else None,
                        "last_error": str(last_error),
                    }
                continue

            last_frame_ts = _now()
            frame_idx += 1

            # --- scale estimation from reference object ---
            ref_info = update_scale_from_reference(frame)

            if len(config.SCALE_BUF) > 0:
                scale_mean = float(np.mean(config.SCALE_BUF))
                scale_std = float(np.std(config.SCALE_BUF))
            else:
                scale_mean = None
                scale_std = None

            # Processed FPS control
            do_infer = (frame_idx % max(1, config.INFER_EVERY_N_FRAMES) == 0)

            if do_infer:
                processed_idx += 1

                # --- DETR inference ---
                x, meta = preprocess_fixed(frame)
                try:
                    logits, pred_boxes = detr.infer(x)
                    boxes_xyxy, scores = postprocess_from_fixed(pred_boxes, logits, meta, conf_thr=config.CONF_THR)
                    last_detr_ts = _now()
                    # clear last_error only on a successful infer (keeps latest meaningful error otherwise)
                    if last_error and str(last_error).startswith("DETR infer failed"):
                        last_error = None
                except Exception as e:
                    err = f"DETR infer failed: {e}"
                    last_error = err
                    print("[live_cam_trt] " + err, flush=True)
                    print(traceback.format_exc(), flush=True)
                    with state.STATE_LOCK:
                        state.LATEST_STATS = {"ready": False, "error": err, "timestamp": _now()}
                    boxes_xyxy = np.zeros((0, 4), np.float32)
                    scores = np.zeros((0,), np.float32)

                if boxes_xyxy.shape[0] > 0:
                    keep = nms_np(boxes_xyxy, scores, iou_thr=config.NMS_IOU)
                    boxes_xyxy = boxes_xyxy[keep]
                    scores = scores[keep]

                # --- compute sizes ---
                L, W, D = compute_LWD(boxes_xyxy)

                if D.size:
                    L_buf.extend(L.tolist())
                    W_buf.extend(W.tolist())
                    D_buf.extend(D.tolist())

                    if len(L_buf) > config.ROLLING_MAX:
                        del L_buf[: len(L_buf) - config.ROLLING_MAX]
                    if len(W_buf) > config.ROLLING_MAX:
                        del W_buf[: len(W_buf) - config.ROLLING_MAX]
                    if len(D_buf) > config.ROLLING_MAX:
                        del D_buf[: len(D_buf) - config.ROLLING_MAX]

                # --- Oversize alarm logic (mm after calibration) ---
                alarm_active = False
                max_d_mm = None
                if config.ALARM_ENABLED and config.PIXELS_PER_MM and D.size > 0:
                    max_d_mm = float(np.max(D))
                    if max_d_mm > config.ALARM_THRESHOLD_MM:
                        alarm_active = True
                config.ALARM_ACTIVE = alarm_active
                config.ALARM_MAX_D_MM = max_d_mm

                # Save last results for overlay on skipped frames
                last_boxes_xyxy = boxes_xyxy.copy()
                last_scores = scores.copy()
                last_D = D.copy()
            else:
                boxes_xyxy = last_boxes_xyxy
                scores = last_scores
                D = last_D

            vis = frame.copy()

            # --- draw reference object ---
            if ref_info is not None:
                cx, cy, diam_px = ref_info
                r = int(diam_px / 2.0)
                cv2.circle(vis, (cx, cy), r, (255, 255, 255), 2)
                cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

            # --- draw boxes with TWO colors only ---
            for (x1, y1, x2, y2), sc, d_val in zip(boxes_xyxy.astype(int), scores, D):
                is_oversized = (config.ALARM_ENABLED and (config.PIXELS_PER_MM is not None) and (d_val > config.ALARM_THRESHOLD_MM))

                color_bgr = (0, 0, 255) if is_oversized else (0, 255, 0)  # RED if oversized else GREEN
                cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, 2)

                # oversized badge (kept)
                if is_oversized:
                    box_w = max(1, x2 - x1)
                    box_h = max(1, y2 - y1)
                    badge_r = int(max(6, min(12, min(box_w, box_h) * 0.18)))
                    cx_badge = x2 - badge_r - 2
                    cy_badge = y1 + badge_r + 2
                    cv2.circle(vis, (cx_badge, cy_badge), badge_r, (0, 0, 255), -1)
                    cv2.putText(
                        vis,
                        "!",
                        (cx_badge - badge_r // 3, cy_badge + badge_r // 2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        lineType=cv2.LINE_AA,
                    )

                if not headless:
                    cv2.putText(vis, f"{sc:.2f}", (x1, max(0, y1 - 6)), 0, 0.5, color_bgr, 1)

            # --- FPS ---
            now = time.time()
            fps_inst = 1.0 / max(1e-3, now - t_prev)
            fps = 0.9 * fps + 0.1 * fps_inst
            t_prev = now
            if not headless:
                cv2.putText(vis, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # =================== Moisture (Top-K crops) ===================
            # Updated: cadence tied to PROCESSED frames (do_infer) to keep it consistent after frame skipping
            if config.MOISTURE_ENABLED and moist and do_infer and (processed_idx % max(1, config.MOISTURE_EVERY_N_FRAMES) == 0):
                try:
                    crops_bhwc, used_boxes = _boxes_to_topk_crops(frame, boxes_xyxy, scores, config.MOISTURE_TOPK)

                    if (crops_bhwc is None or len(used_boxes) == 0) and config.MOISTURE_FALLBACK_FULLFRAME:
                        crops_bhwc = np.stack([_prep_moist_crop_bgr(frame, config.MOISTURE_INPUT_SIZE)], axis=0).astype(np.float32)
                        used_boxes = []

                    if crops_bhwc is not None:
                        probs = moist.infer_probs_bhwc(crops_bhwc)
                        if probs is not None:
                            mean_probs, pred_idx, pred_label = _moisture_aggregate(probs)
                            last_moist_ts = _now()
                            if last_error and str(last_error).startswith("Moisture infer failed"):
                                last_error = None

                            classes = moisture_mod.MOISTURE_CLASSES
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
                                        )
                                    })

                            with state.STATE_LOCK:
                                state.LATEST_MOISTURE = {
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
                                    "topk": int(config.MOISTURE_TOPK),
                                    "boxes_used": int(len(used_boxes)),
                                    "per_box": per_box,
                                }
                    else:
                        with state.STATE_LOCK:
                            state.LATEST_MOISTURE = {
                                "ready": False,
                                "timestamp": _now(),
                                "error": "No crops available for moisture (no boxes and fallback disabled).",
                                "topk": int(config.MOISTURE_TOPK),
                                "boxes_used": 0,
                            }
                except Exception as e:
                    err = f"Moisture infer failed: {e}"
                    last_error = err
                    print("[live_cam_trt] " + err, flush=True)
                    print(traceback.format_exc(), flush=True)
                    with state.STATE_LOCK:
                        state.LATEST_MOISTURE = {"ready": False, "timestamp": _now(), "error": err}

            # =================== Update SHARED STATE ===================
            stats_dict = compute_diameter_stats(D_buf, config.UNITS, scale_mean=scale_mean, scale_std=scale_std)
            if stats_dict is not None:
                stats_dict["alarm_active"] = bool(config.ALARM_ACTIVE)
                stats_dict["alarm_threshold_mm"] = float(config.ALARM_THRESHOLD_MM)
                stats_dict["alarm_max_d_mm"] = float(config.ALARM_MAX_D_MM) if config.ALARM_MAX_D_MM is not None else None
                stats_dict["ref_diam_mm"] = float(config.REF_DIAM_MM)

            # histogram package + threshold line info (UPDATED: diameter only, throttled to 1 Hz)
            hist_data = None
            should_update_hist = (now - last_hist_time) >= float(config.HIST_UPDATE_INTERVAL)

            if should_update_hist and len(D_buf) > 10:
                last_hist_time = now

                def make_diameter_hist(values):
                    vals = np.array(values, dtype=np.float32)
                    if len(vals) < 5:
                        return None

                    # Fixed range when calibrated (mm); percentile range when still in pixels
                    if config.UNITS == "mm":
                        upper = float(config.HIST_D_MAX_MM)
                        upper = max(1.0, upper)
                        edges = np.linspace(0.0, upper, config.HIST_NUM_BINS + 1, dtype=np.float32)
                    else:
                        upper = float(np.percentile(vals, 99.0))
                        if upper <= 0:
                            upper = float(np.max(vals)) if len(vals) > 0 else 1.0
                        upper = max(1.0, upper)
                        edges = np.linspace(0.0, upper, config.HIST_NUM_BINS + 1, dtype=np.float32)

                    hist, _ = np.histogram(vals, bins=edges)
                    centers = 0.5 * (edges[:-1] + edges[1:])

                    nonlocal prev_hist_counts, prev_hist_bins  # type: ignore
                    if config.HIST_EMA_ENABLED:
                        if prev_hist_counts is not None and prev_hist_bins is not None and len(prev_hist_counts) == len(hist) and np.allclose(prev_hist_bins, centers):
                            sm = (1.0 - config.HIST_EMA_ALPHA) * prev_hist_counts + config.HIST_EMA_ALPHA * hist.astype(np.float32)
                            hist_out = np.rint(sm).astype(int)
                        else:
                            hist_out = hist.astype(int)

                        prev_hist_counts = hist_out.astype(np.float32)
                        prev_hist_bins = centers.copy()
                    else:
                        hist_out = hist.astype(int)

                    return {"bins": centers.tolist(), "counts": hist_out.tolist()}

                h_dia = make_diameter_hist(D_buf)

                threshold_x = None
                pct_over_threshold = None
                if config.ALARM_ENABLED and config.UNITS == "mm" and len(D_buf) > 0:
                    thr = float(config.ALARM_THRESHOLD_MM)
                    threshold_x = thr
                    darr = np.array(D_buf, dtype=np.float32)
                    pct_over_threshold = float(100.0 * np.mean(darr > thr))

                hist_data = {
                    "ready": True,
                    "timestamp": _now(),
                    "units": config.UNITS,
                    "mode": "diameter",
                    "diameter": h_dia,

                    "threshold_x": threshold_x,
                    "pct_over_threshold": pct_over_threshold,
                    "alarm_threshold_mm": float(config.ALARM_THRESHOLD_MM),
                    "alarm_enabled": bool(config.ALARM_ENABLED),

                    "hist_update_interval": float(config.HIST_UPDATE_INTERVAL),
                    "hist_num_bins": int(config.HIST_NUM_BINS),
                    "hist_d_max_mm": float(config.HIST_D_MAX_MM) if config.UNITS == "mm" else None,
                    "hist_ema_enabled": bool(config.HIST_EMA_ENABLED),
                    "hist_ema_alpha": float(config.HIST_EMA_ALPHA) if config.HIST_EMA_ENABLED else None,
                }
                last_hist_pub_ts = _now()

            # --- publish shared state (frame/stats/hist/health) ---
            health_dict = {
                "ready": True,
                "timestamp": _now(),
                "uptime_sec": float(_now() - start_ts),
                "camera_ok": True,
                "cam_dev": str(config.CAM_DEV),

                "last_frame_ts": last_frame_ts,
                "last_detr_ts": last_detr_ts,
                "last_moist_ts": last_moist_ts,
                "last_hist_pub_ts": last_hist_pub_ts,

                "fps_smoothed": float(fps),
                "frame_idx": int(frame_idx),
                "processed_idx": int(processed_idx),
                "infer_every_n_frames": int(config.INFER_EVERY_N_FRAMES),

                "units": str(config.UNITS),
                "alarm_active": bool(config.ALARM_ACTIVE),
                "alarm_threshold_mm": float(config.ALARM_THRESHOLD_MM),
                "alarm_max_d_mm": float(config.ALARM_MAX_D_MM) if config.ALARM_MAX_D_MM is not None else None,

                "last_error": str(last_error) if last_error else None,
            }

            with state.STATE_LOCK:
                state.LATEST_FRAME = vis.copy()
                state.LATEST_STATS = stats_dict if stats_dict is not None else {"ready": False, "timestamp": _now(), "error": "No detections yet."}
                if hist_data is not None:
                    state.HIST_DATA = hist_data
                # else keep last HIST_DATA to avoid blanking between 1Hz updates

                state.LATEST_HEALTH = health_dict

            # Debug GUI
            if not headless:
                disp = draw_stats_panel(vis.copy(), D_buf, config.UNITS, scale_mean=scale_mean, scale_std=scale_std)
                disp = draw_color_legend(disp)
                cv2.imshow("woodchip-live", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.001)

        print("[live_cam_trt] Exiting inference loop.", flush=True)

    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            del moist
        except Exception:
            pass
        try:
            del detr
        except Exception:
            pass

        try:
            ctx.pop()
        except Exception:
            pass


def main():
    run_inference_loop(headless=False)


if __name__ == "__main__":
    main()
