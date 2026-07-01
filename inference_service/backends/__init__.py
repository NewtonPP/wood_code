# inference_service.backends — pluggable model backends.
"""
A ``ModelBackend`` turns a raw BGR frame into DETR boxes/scores and a batch of
crops into moisture probabilities. The post-processing (sizing, calibration,
alarm, histogram, moisture aggregation) is provided once by
``woodchip_core.FrameProcessor`` regardless of backend.

Select the backend with the ``INFER_BACKEND`` env var (default ``mock``).
"""

import os

from .base import ModelBackend


def make_backend(name=None) -> ModelBackend:
    name = (name or os.environ.get("INFER_BACKEND", "mock")).strip().lower()
    if name in ("mock", "fake", ""):
        from .mock_backend import MockBackend
        return MockBackend()
    if name == "onnx":
        from .onnx_backend import OnnxBackend
        return OnnxBackend()
    raise ValueError(f"Unknown INFER_BACKEND={name!r} (expected 'mock' or 'onnx')")
