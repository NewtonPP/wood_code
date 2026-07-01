# inference_service/main.py
"""
Cloud inference service (private; only the backend calls it).

POST /infer  -> run the model + woodchip_core post-processing for one frame of
               one device, returning the full JSON payload the frontend consumes.
GET  /healthz
POST /config -> propagate operator rule changes (conf/nms/alarm/moisture, …)
               from the backend to the shared post-processing config.

A ``FrameProcessor`` is kept per ``device_id`` so each camera stream carries its
own calibration / rolling buffers / histogram state.
"""

import base64
import threading
from typing import Dict, Optional

import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from woodchip_core import config as core_config
from woodchip_core import update_runtime_config
from woodchip_core.processor import FrameProcessor

from .backends import make_backend

app = FastAPI(title="Woodchip Inference Service")

BACKEND = make_backend()

_processors: Dict[str, FrameProcessor] = {}
_lock = threading.Lock()


class InferRequest(BaseModel):
    device_id: str
    frame_jpeg_b64: str


def _decode_frame(b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}")
    buf = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="could not decode JPEG frame")
    return frame


def _get_processor(device_id: str) -> FrameProcessor:
    with _lock:
        proc = _processors.get(device_id)
        if proc is None:
            proc = FrameProcessor(
                cfg=core_config,
                moisture_classes=BACKEND.moisture_classes,
                initial_pixels_per_mm=BACKEND.default_pixels_per_mm,
            )
            _processors[device_id] = proc
        return proc


@app.get("/healthz")
def healthz():
    return {"ok": True, "backend": BACKEND.name, "devices": len(_processors)}


@app.post("/infer")
def infer(req: InferRequest):
    frame = _decode_frame(req.frame_jpeg_b64)
    proc = _get_processor(req.device_id)
    boxes, scores = BACKEND.infer_detr(frame)
    return proc.process(frame, boxes, scores, moisture_infer=BACKEND.infer_moisture)


class ConfigUpdate(BaseModel):
    conf_thr: Optional[float] = None
    nms_iou: Optional[float] = None
    alarm_threshold_mm: Optional[float] = None
    alarm_enabled: Optional[bool] = None
    ref_diam_mm: Optional[float] = None
    histogram_mode: Optional[str] = None
    moisture_enabled: Optional[bool] = None
    moisture_topk: Optional[int] = None
    moisture_every_n_frames: Optional[int] = None


@app.get("/config")
def get_config():
    return core_config.get_runtime_config()


@app.post("/config")
def post_config(update: ConfigUpdate):
    update_runtime_config(**update.dict(exclude_none=True))
    return core_config.get_runtime_config()
