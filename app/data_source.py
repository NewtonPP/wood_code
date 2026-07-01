# app/data_source.py
"""
Indirection that returns live inference data from the right source for the
current deployment role, so the live/devices routers and the event sampler do
not need to know whether they're on a Jetson or in the cloud.

- device role: read ``live_cam_trt``'s in-process shared state (single camera).
- cloud  role: read the per-device :mod:`app.live_store`, scoped to the
  authenticated user's own device (tenant isolation).

v1 is single-camera-per-account, so a user's ``device_id`` is derived from their
account id. A future devices registry can replace ``device_id_for_user`` without
touching the routers.
"""

from typing import Optional

from . import config


def device_id_for_user(user: dict) -> str:
    """Stable per-account device id (v1: one camera per account)."""
    return f"user-{user.get('id')}"


# ---- device-role accessors (lazy import keeps live_cam_trt out of cloud hot path) ----
def _live_cam():
    import live_cam_trt  # safe to import: __init__ pulls no CUDA/TensorRT
    return live_cam_trt


# ---- live reads (role-aware, tenant-scoped) ----
def get_stats(user: dict):
    if config.ROLE == "cloud":
        from . import live_store
        return live_store.get_stats(device_id_for_user(user))
    return _live_cam().get_latest_stats()


def get_hist(user: dict):
    if config.ROLE == "cloud":
        from . import live_store
        return live_store.get_hist(device_id_for_user(user))
    return _live_cam().get_hist_data()


def get_moisture(user: dict):
    if config.ROLE == "cloud":
        from . import live_store
        return live_store.get_moisture(device_id_for_user(user))
    return _live_cam().get_latest_moisture()


def get_health(user: dict):
    if config.ROLE == "cloud":
        from . import live_store
        return live_store.get_health(device_id_for_user(user))
    return _live_cam().get_latest_health()


def get_frame(user: dict):
    """Annotated server frame. Cloud renders overlays in the browser, so there is
    no server-side frame there — return None (the endpoint replies 204)."""
    if config.ROLE == "cloud":
        return None
    return _live_cam().get_latest_frame()
