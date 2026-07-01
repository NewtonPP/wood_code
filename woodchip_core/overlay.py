# woodchip_core/overlay.py
"""
Statistics computation + optional on-frame UI rendering (stats panel, legend).

``classify_batch`` / ``compute_diameter_stats`` are pure numpy and used by both
the device and the cloud. ``draw_stats_panel`` / ``draw_color_legend`` use cv2
for the optional debug GUI on the device; the cloud does not draw (the browser
renders overlays from the returned JSON).
"""

import numpy as np
import cv2


def classify_batch(D_buf):
    if len(D_buf) < 10:
        return "Collecting..."
    std = np.std(D_buf)
    if std < 30:
        return "Uniform"
    elif std < 60:
        return "Moderate"
    else:
        return "Diverse"


def compute_diameter_stats(D_buf, units, scale_mean=None, scale_std=None):
    if len(D_buf) == 0:
        return None
    D_arr = np.array(D_buf, dtype=np.float32)
    stats = {
        "ready": True,
        "mean_d": float(np.mean(D_arr)),
        "median_d": float(np.median(D_arr)),
        "std_d": float(np.std(D_arr)),
        "min_d": float(np.min(D_arr)),
        "max_d": float(np.max(D_arr)),
        "batch_label": classify_batch(D_buf),
        "units": units,
        "px_per_mm_mean": float(scale_mean) if scale_mean is not None else None,
        "px_per_mm_std": float(scale_std) if scale_std is not None else None,
    }
    return stats


def draw_stats_panel(frame, D_buf, units, scale_mean=None, scale_std=None):
    h, w = frame.shape[:2]
    panel_w, panel_h = 280, 190
    x0, y0 = w - panel_w - 10, 30
    panel = np.zeros((panel_h, panel_w, 3), np.uint8)
    panel[:] = (20, 20, 20)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 25
    cv2.putText(panel, "Diameter Stats", (10, y), font, 0.8, (0, 255, 255), 2)

    if len(D_buf) > 0:
        D_arr = np.array(D_buf, dtype=np.float32)
        mean = float(np.mean(D_arr))
        med = float(np.median(D_arr))
        std = float(np.std(D_arr))
        dmin = float(np.min(D_arr))
        dmax = float(np.max(D_arr))
        batch = classify_batch(D_buf)

        y += 25
        cv2.putText(panel, f"Mean : {mean:.1f} {units}", (10, y), font, 0.7, (0, 255, 0), 1)
        y += 22
        cv2.putText(panel, f"Median: {med:.1f} {units}", (10, y), font, 0.7, (0, 255, 0), 1)
        y += 22
        cv2.putText(panel, f"Std  : {std:.1f} {units}", (10, y), font, 0.7, (0, 255, 0), 1)
        y += 22
        cv2.putText(panel, f"Min  : {dmin:.1f} {units}", (10, y), font, 0.7, (0, 255, 0), 1)
        y += 22
        cv2.putText(panel, f"Max  : {dmax:.1f} {units}", (10, y), font, 0.7, (0, 255, 0), 1)
        y += 22
        cv2.putText(panel, f"Batch: {batch}", (10, y), font, 0.7, (255, 200, 0), 1)

        if scale_mean is not None and scale_mean > 0:
            y += 22
            if scale_std is not None:
                cv2.putText(panel, f"px/mm: {scale_mean:.1f} +/- {scale_std:.1f}", (10, y), font, 0.7, (200, 200, 255), 1)
            else:
                cv2.putText(panel, f"px/mm: {scale_mean:.1f}", (10, y), font, 0.7, (200, 200, 255), 1)
    else:
        y += 50
        cv2.putText(panel, "No data yet...", (10, y), font, 0.7, (0, 0, 255), 1)

    y1, y2 = y0, y0 + panel_h
    x1, x2 = x0, x0 + panel_w
    frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.4, panel, 0.6, 0)
    return frame


def draw_color_legend(frame):
    h, w = frame.shape[:2]
    legend = np.zeros((30, 300, 3), np.uint8)
    legend[:] = (40, 40, 40)

    legend[:, 0:140] = (0, 200, 0)
    cv2.putText(legend, "Normal", (10, 20), 0, 0.55, (0, 0, 0), 1)

    legend[:, 140:160] = (40, 40, 40)
    cv2.putText(legend, "|", (148, 20), 0, 0.55, (255, 255, 255), 1)

    legend[:, 160:300] = (0, 0, 255)
    cv2.putText(legend, "Oversize", (170, 20), 0, 0.55, (255, 255, 255), 1)

    frame[h - 40 : h - 10, 10:310] = legend
    return frame
