# inference_service/model.py
"""
ONNX Runtime models for the inference service.

DETR (chip detection/sizing) is required and loads at startup. The MoistNetLite
moisture model is optional — export it with export_moistnet_onnx.py; if the
file is absent the moisture session is skipped, ``infer_moisture`` returns
``None``, and the UI shows "Collecting…" for moisture.

Model paths (env-overridable; defaults point into ``inference_service/models/``):

    DETR_ONNX_PATH        default models/detr_resnet101.onnx      (required)
    MOISTURE_ONNX_PATH    default models/moistnetlite.onnx        (optional)
    MOISTURE_CLASSES_PATH default models/moistnetlite_classes.txt (one label
                          per line; dry/medium/wet)

Pre/post-processing reuses ``woodchip_core.geometry`` so boxes are identical to
the Jetson TensorRT path. On a GPU host install ``onnxruntime-gpu``; the CUDA
execution provider is picked up automatically when available.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from woodchip_core import config as core_config
from woodchip_core.geometry import preprocess_fixed, postprocess_from_fixed

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _load_classes(path: Optional[str]) -> Optional[List[str]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    return labels or None


class OnnxModel:
    name = "onnx"
    # Real models measure in pixels until the blue reference disk (or a manual
    # DEFAULT_PIXELS_PER_MM) provides the mm scale.
    default_pixels_per_mm: Optional[float] = None

    def __init__(self):
        detr_path = os.environ.get(
            "DETR_ONNX_PATH", os.path.join(_MODELS_DIR, "detr_resnet101.onnx")
        )
        if not os.path.exists(detr_path):
            raise RuntimeError(
                f"DETR model not found at {detr_path!r}. Export it with "
                "inference_service/export_detr_onnx.py (see DEPLOYMENT.md) or "
                "point DETR_ONNX_PATH at the file."
            )

        providers = ort.get_available_providers()
        self._detr = ort.InferenceSession(detr_path, providers=providers)

        moist_path = os.environ.get(
            "MOISTURE_ONNX_PATH", os.path.join(_MODELS_DIR, "moistnetlite.onnx")
        )
        self._moist = (
            ort.InferenceSession(moist_path, providers=providers)
            if os.path.exists(moist_path)
            else None
        )
        self.moisture_classes = _load_classes(
            os.environ.get(
                "MOISTURE_CLASSES_PATH",
                os.path.join(_MODELS_DIR, "moistnetlite_classes.txt"),
            )
        )

    @property
    def moisture_loaded(self) -> bool:
        return self._moist is not None

    def infer_detr(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (boxes_xyxy float32 [N,4], scores float32 [N]) in frame pixels."""
        x, meta = preprocess_fixed(frame_bgr)
        inp = self._detr.get_inputs()[0].name
        logits, pred_boxes = self._detr.run(None, {inp: x})
        return postprocess_from_fixed(pred_boxes, logits, meta, conf_thr=core_config.CONF_THR)

    def infer_moisture(self, crops_bhwc: np.ndarray) -> Optional[np.ndarray]:
        """Return per-crop class probabilities float32 [N, C], or None if no model."""
        if self._moist is None:
            return None
        # The model may expect NHWC (Keras/tf2onnx export) or NCHW; core produces NHWC.
        moist_input = self._moist.get_inputs()[0]
        shape = moist_input.shape
        if len(shape) == 4 and shape[-1] == 3:
            x = crops_bhwc.astype(np.float32)
        else:
            x = np.transpose(crops_bhwc, (0, 3, 1, 2)).astype(np.float32)
        out = self._moist.run(None, {moist_input.name: x})[0]
        out = np.asarray(out, np.float32)
        # Softmax if the model emits logits rather than probabilities.
        if not np.allclose(out.sum(axis=1), 1.0, atol=1e-2):
            ex = np.exp(out - out.max(axis=1, keepdims=True))
            out = ex / ex.sum(axis=1, keepdims=True)
        return out
