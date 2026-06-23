# app/routers/live.py
"""
Protected live API: stats / frame / hist / moisture.
"""

import cv2
from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse

import live_cam_trt  # owns CUDA + TensorRT

from ..auth import require_perm

router = APIRouter()


@router.get("/api/stats")
def get_stats(user: dict = Depends(require_perm("view_live"))):
    stats = live_cam_trt.get_latest_stats()
    if stats is None:
        return JSONResponse({"ready": False})
    return JSONResponse(stats)


@router.get("/api/frame")
def get_frame(user: dict = Depends(require_perm("view_live"))):
    frame = live_cam_trt.get_latest_frame()
    if frame is None:
        return Response(status_code=204)

    ok, jpg = cv2.imencode(".jpg", frame)
    if not ok:
        return Response(status_code=500)

    return Response(content=jpg.tobytes(), media_type="image/jpeg")


@router.get("/api/hist")
def get_hist(user: dict = Depends(require_perm("view_live"))):
    hist = live_cam_trt.get_hist_data()
    if not hist:
        return JSONResponse({"ready": False})

    if isinstance(hist, dict) and hist.get("ready") and hist.get("mode") == "diameter":
        dia = hist.get("diameter")
        if not (isinstance(dia, dict) and "bins" in dia and "counts" in dia):
            safe = dict(hist)
            safe["ready"] = False
            safe["error"] = "Histogram payload missing diameter bins/counts."
            return JSONResponse(safe)

    return JSONResponse(hist)


@router.get("/api/moisture")
def get_moisture(user: dict = Depends(require_perm("view_live"))):
    moist = live_cam_trt.get_latest_moisture()
    if not moist:
        return JSONResponse({"ready": False})
    return JSONResponse(moist)
