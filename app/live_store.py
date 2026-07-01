# app/live_store.py
"""
Per-device live-data store for the cloud role.

The Jetson build keeps live data in ``live_cam_trt``'s module globals because
there is exactly one camera in one process. In the cloud, many browsers stream
to one (or more) backend instances, so the latest stats/histogram/moisture/
health are kept here keyed by ``device_id``.

This is a deliberately tiny in-memory implementation. It is the single seam
where a shared store (**Redis / Memorystore**) plugs in when the backend scales
to multiple instances — replace the ``_STORE`` dict accesses with Redis
GET/SETs and the rest of the app is unchanged.
"""

import threading
import time
from typing import Dict, List, Optional

_LOCK = threading.Lock()
# device_id -> {"stats":..., "hist":..., "moisture":..., "health":..., "updated_at": float}
_STORE: Dict[str, dict] = {}

# Devices are considered "online" if they pushed a frame within this window.
STALE_AFTER_SEC = 10.0


def update(device_id: str, *, stats=None, hist=None, moisture=None, health=None) -> None:
    with _LOCK:
        rec = _STORE.setdefault(device_id, {})
        if stats is not None:
            rec["stats"] = stats
        if hist is not None:
            rec["hist"] = hist
        if moisture is not None:
            rec["moisture"] = moisture
        if health is not None:
            rec["health"] = health
        rec["updated_at"] = time.time()


def _field(device_id: str, key: str):
    with _LOCK:
        rec = _STORE.get(device_id)
        return rec.get(key) if rec else None


def get_stats(device_id: str):
    return _field(device_id, "stats")


def get_hist(device_id: str):
    return _field(device_id, "hist")


def get_moisture(device_id: str):
    return _field(device_id, "moisture")


def get_health(device_id: str):
    return _field(device_id, "health")


def is_online(device_id: str) -> bool:
    with _LOCK:
        rec = _STORE.get(device_id)
        return bool(rec) and (time.time() - rec.get("updated_at", 0)) <= STALE_AFTER_SEC


def active_device_ids() -> List[str]:
    """device_ids seen within the staleness window (used by the event sampler)."""
    now = time.time()
    with _LOCK:
        return [d for d, rec in _STORE.items() if (now - rec.get("updated_at", 0)) <= STALE_AFTER_SEC]
