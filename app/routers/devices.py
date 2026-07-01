# app/routers/devices.py
"""
Device health status (role-aware via app.data_source).
"""

from fastapi import APIRouter, Depends

from ..auth import require_perm
from ..time_utils import utc_now_iso
from ..db import list_devices
from .. import data_source

router = APIRouter()


@router.get("/api/devices/status")
def device_status(user: dict = Depends(require_perm("view_devices"))):
    health = data_source.get_health(user)
    if not health:
        # fallback: at least tell if stats exist
        stats = data_source.get_stats(user)
        return {"ready": False, "timestamp": utc_now_iso(), "stats_ready": bool(stats)}
    return health


@router.get("/api/devices")
def devices_list(user: dict = Depends(require_perm("view_devices"))):
    """Device inventory (registry). Oversight roles (with view_devices) see all
    registered devices and their owners + last-seen time."""
    return {"ok": True, "devices": list_devices()}
