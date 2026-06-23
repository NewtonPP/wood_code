# app/routers/devices.py
"""
Device health status.
"""

from fastapi import APIRouter, Depends

import live_cam_trt  # owns CUDA + TensorRT

from ..auth import require_perm
from ..time_utils import utc_now_iso

router = APIRouter()


@router.get("/api/devices/status")
def device_status(user: dict = Depends(require_perm("view_devices"))):
    health = live_cam_trt.get_latest_health()
    if not health:
        # fallback: at least tell if stats exist
        stats = live_cam_trt.get_latest_stats()
        return {"ready": False, "timestamp": utc_now_iso(), "stats_ready": bool(stats)}
    return health
