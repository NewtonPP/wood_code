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
import os
import threading
from typing import Dict, Optional

import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from woodchip_core import config as core_config
from woodchip_core import update_runtime_config
from woodchip_core.processor import FrameProcessor

from .model import OnnxModel

app = FastAPI(title="Woodchip Inference Service")

MODEL = OnnxModel()

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


# Manual calibration for fixed-mount cameras: if the blue reference disk is
# not (always) in view, set DEFAULT_PIXELS_PER_MM so sizing/alarms run in mm
# immediately. When the disk IS detected it refines this scale online.
_DEFAULT_PPM = float(os.environ.get("DEFAULT_PIXELS_PER_MM", "0")) or None


def _get_processor(device_id: str) -> FrameProcessor:
    with _lock:
        proc = _processors.get(device_id)
        if proc is None:
            proc = FrameProcessor(
                cfg=core_config,
                moisture_classes=MODEL.moisture_classes,
                initial_pixels_per_mm=MODEL.default_pixels_per_mm or _DEFAULT_PPM,
            )
            _processors[device_id] = proc
        return proc


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "model": MODEL.name,
        "moisture_loaded": MODEL.moisture_loaded,
        "devices": len(_processors),
    }


@app.post("/infer")
def infer(req: InferRequest):
    frame = _decode_frame(req.frame_jpeg_b64)
    proc = _get_processor(req.device_id)
    boxes, scores = MODEL.infer_detr(frame)
    return proc.process(frame, boxes, scores, moisture_infer=MODEL.infer_moisture)


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
