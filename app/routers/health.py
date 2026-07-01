# app/routers/health.py
"""
Public health check + device info.
"""

from fastapi import APIRouter

from .. import config

router = APIRouter()


@router.get("/ping")
def ping():
    return {"status": "ok"}


@router.get("/api/info")
def info():
    """Public: identifies this instance and its deployment role.

    ``role`` tells the frontend how to source the live feed: "cloud" -> the
    browser captures the camera and streams to the ingest WebSocket; "device"
    -> the server provides annotated frames at /api/frame.
    """
    return {"ok": True, "device_id": config.DEVICE_ID, "role": config.ROLE}
