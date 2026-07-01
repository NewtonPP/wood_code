# inference_service/backends/onnx_backend.py
"""
ONNX Runtime backend (inactive until model files exist).

Wiring is in place so that, once the DETR + moisture models are exported to
ONNX, serving is a matter of pointing these env vars at the files:

    INFER_BACKEND=onnx
    DETR_ONNX_PATH=/models/detr_resnet101.onnx
    MOISTURE_ONNX_PATH=/models/moistnetlite.onnx
    MOISTURE_CLASSES_PATH=/models/moisture_classes.txt   # one label per line

Preprocess/postprocess reuse ``woodchip_core.geometry`` so the boxes are
identical to the Jetson path. On a GPU host install ``onnxruntime-gpu``; the
CUDA execution provider is used automatically when available.
"""

import os
from typing import List, Optional, Tuple

import numpy as np

from woodchip_core import config as core_config
from woodchip_core.geometry import preprocess_fixed, postprocess_from_fixed

from .base import ModelBackend


def _load_classes(path: Optional[str]) -> Optional[List[str]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    return labels or None


class OnnxBackend(ModelBackend):
    default_pixels_per_mm = None  # real models calibrate from the reference disk

    def __init__(self):
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on deploy image
            raise RuntimeError(
                "onnxruntime is not installed. Install 'onnxruntime-gpu' (GPU host) "
                "or 'onnxruntime' (CPU) to use INFER_BACKEND=onnx."
            ) from e

        detr_path = os.environ.get("DETR_ONNX_PATH")
        moist_path = os.environ.get("MOISTURE_ONNX_PATH")
        if not detr_path or not os.path.exists(detr_path):
            raise RuntimeError(
                f"DETR_ONNX_PATH not set or missing ({detr_path!r}). Export the DETR "
                "model to ONNX and point DETR_ONNX_PATH at it."
            )

        providers = ort.get_available_providers()
        self._detr = ort.InferenceSession(detr_path, providers=providers)
        self._moist = None
        if moist_path and os.path.exists(moist_path):
            self._moist = ort.InferenceSession(moist_path, providers=providers)

        self.moisture_classes = _load_classes(os.environ.get("MOISTURE_CLASSES_PATH"))

    def infer_detr(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, meta = preprocess_fixed(frame_bgr)
        inp = self._detr.get_inputs()[0].name
        logits, pred_boxes = self._detr.run(None, {inp: x})
        return postprocess_from_fixed(pred_boxes, logits, meta, conf_thr=core_config.CONF_THR)

    def infer_moisture(self, crops_bhwc: np.ndarray) -> Optional[np.ndarray]:
        if self._moist is None:
            return None
        # Models typically expect NCHW; transpose from the NHWC crops core produces.
        x = np.transpose(crops_bhwc, (0, 3, 1, 2)).astype(np.float32)
        inp = self._moist.get_inputs()[0].name
        out = self._moist.run(None, {inp: x})[0]
        out = np.asarray(out, np.float32)
        # Softmax if the model emits logits rather than probabilities.
        if not np.allclose(out.sum(axis=1), 1.0, atol=1e-2):
            ex = np.exp(out - out.max(axis=1, keepdims=True))
            out = ex / ex.sum(axis=1, keepdims=True)
        return out
