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
    """Public: identifies which Jetson/appliance this instance is."""
    return {"ok": True, "device_id": config.DEVICE_ID}
