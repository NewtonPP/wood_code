# app/sampler.py
"""
Lightweight background sampler: snapshots runtime telemetry into the events DB
(~1 row/sec). Does not touch CUDA.

Role-aware:
- device: snapshot live_cam_trt's single-camera shared state -> one row/sec.
- cloud:  snapshot each online device in the per-device live store -> one row
          per device/sec, tagged with that device's id.
"""

import json
import time
from typing import Any

from . import config
from .db import _DB_LOCK, db_connect, audit_log
from .time_utils import utc_now_iso, utc_now_epoch


def _safe_get(d: Any, key: str, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def _write_event(device_id, stats, moist, health):
    ts_iso = utc_now_iso()
    ts_epoch = utc_now_epoch()

    alarm_active = int(bool(_safe_get(stats, "alarm_active", False))) if isinstance(stats, dict) else 0
    alarm_max_d_mm = _safe_get(stats, "alarm_max_d_mm", None) if isinstance(stats, dict) else None
    units = _safe_get(stats, "units", None) or _safe_get(stats, "UNITS", None)
    mean_d = _safe_get(stats, "mean", None) or _safe_get(stats, "mean_d", None) or _safe_get(stats, "mean_diameter", None)
    std_d = _safe_get(stats, "std", None) or _safe_get(stats, "std_d", None) or _safe_get(stats, "std_diameter", None)

    moisture_mean_pred = None
    if isinstance(moist, dict) and moist.get("ready"):
        moisture_mean_pred = moist.get("mean_pred")

    payload = {
        "stats": stats if isinstance(stats, dict) else None,
        "moisture": moist if isinstance(moist, dict) else None,
        "health": health if isinstance(health, dict) else None,
    }

    with _DB_LOCK:
        conn = db_connect()
        conn.execute("""
            INSERT INTO events
            (ts, ts_epoch, device_id, alarm_active, alarm_max_d_mm, mean_d, std_d, units, moisture_mean_pred, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts_iso,
            float(ts_epoch),
            device_id,
            int(alarm_active),
            float(alarm_max_d_mm) if alarm_max_d_mm is not None else None,
            float(mean_d) if mean_d is not None else None,
            float(std_d) if std_d is not None else None,
            str(units) if units is not None else None,
            str(moisture_mean_pred) if moisture_mean_pred is not None else None,
            json.dumps(payload, ensure_ascii=False),
        ))
        conn.commit()
        conn.close()


def _sample_once():
    if config.ROLE == "cloud":
        from . import live_store
        for device_id in live_store.active_device_ids():
            _write_event(
                device_id,
                live_store.get_stats(device_id),
                live_store.get_moisture(device_id),
                live_store.get_health(device_id),
            )
    else:
        import live_cam_trt
        _write_event(
            config.DEVICE_ID,
            live_cam_trt.get_latest_stats(),
            live_cam_trt.get_latest_moisture(),
            live_cam_trt.get_latest_health(),
        )


def _event_sampler_loop():
    """Periodically snapshot runtime telemetry -> events DB. Does not touch CUDA."""
    while True:
        try:
            _sample_once()
        except Exception as e:
            # Don't crash sampler
            audit_log("event_sampler_error", None, {"error": str(e)})
        time.sleep(1.0)
