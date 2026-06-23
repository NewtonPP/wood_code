# live_cam_trt/state.py
"""
Thread-safe shared state published by the inference loop and consumed by the
FastAPI backend.

The inference loop writes these by assigning ``state.LATEST_* = ...`` under
``STATE_LOCK``. Consumers read them through the accessor functions below so that
they always observe the latest reassigned value (importing the names directly
would bind a stale reference at import time).
"""

import threading


STATE_LOCK = threading.Lock()

LATEST_FRAME = None     # BGR np.uint8 image (annotated)
LATEST_STATS = None     # dict
HIST_DATA = None        # dict
LATEST_MOISTURE = None  # dict
LATEST_HEALTH = None


def get_health():
    with STATE_LOCK:
        return LATEST_HEALTH.copy() if isinstance(LATEST_HEALTH, dict) else None


# --- Live accessors (read current value at call time) ---
def get_latest_frame():
    return LATEST_FRAME


def get_latest_stats():
    return LATEST_STATS


def get_hist_data():
    return HIST_DATA


def get_latest_moisture():
    return LATEST_MOISTURE


def get_latest_health():
    return LATEST_HEALTH
