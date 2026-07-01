# app/routers/live.py
"""
Protected live API: stats / frame / hist / moisture.

Reads through ``app.data_source`` so the same endpoints work in both roles:
- device: live_cam_trt in-process shared state.
- cloud:  the per-device live store (scoped to the authenticated user).
"""

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse

from ..auth import require_perm
from .. import data_source

router = APIRouter()


@router.get("/api/stats")
def get_stats(user: dict = Depends(require_perm("view_live"))):
    stats = data_source.get_stats(user)
    if stats is None:
        return JSONResponse({"ready": False})
    return JSONResponse(stats)


@router.get("/api/frame")
def get_frame(user: dict = Depends(require_perm("view_live"))):
    frame = data_source.get_frame(user)
    if frame is None:
        # Cloud renders overlays in the browser; there is no server frame.
        return Response(status_code=204)

    import cv2  # device-only path; keeps OpenCV out of the cloud import graph
    ok, jpg = cv2.imencode(".jpg", frame)
    if not ok:
        return Response(status_code=500)

    return Response(content=jpg.tobytes(), media_type="image/jpeg")


@router.get("/api/hist")
def get_hist(user: dict = Depends(require_perm("view_live"))):
    hist = data_source.get_hist(user)
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
    moist = data_source.get_moisture(user)
    if not moist:
        return JSONResponse({"ready": False})
    return JSONResponse(moist)
