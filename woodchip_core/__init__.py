# woodchip_core/__init__.py
"""
woodchip_core — the platform-independent (CUDA-free) wood-chip post-processing.

Everything here is pure numpy + cv2: sizing/calibration, alarm, histogram, and
moisture aggregation. It has **no** TensorRT/pycuda/Jetson dependency, so it can
run in the cloud inference service, on a dev Mac, or on the Jetson device.

The Jetson package ``live_cam_trt`` re-exports these implementations (see the
thin shims in ``live_cam_trt/{geometry,reference,moisture,overlay}.py``) so the
device and the cloud share exactly one post-processing code path.

Note: ``FrameProcessor`` (and the cv2-dependent helpers) are loaded lazily via
``__getattr__`` so that merely importing ``woodchip_core`` — e.g. the cloud
*backend* using only :func:`get_runtime_config` / :func:`update_runtime_config`
— does not require OpenCV. Only the inference service (which decodes frames)
pulls cv2 in, by importing :class:`woodchip_core.processor.FrameProcessor`.
"""

from . import config  # noqa: F401  (numpy-only, no cv2)
from .config import get_runtime_config, update_runtime_config  # noqa: F401

__all__ = [
    "config",
    "get_runtime_config",
    "update_runtime_config",
    "FrameProcessor",
]


def __getattr__(name):  # PEP 562 lazy attribute
    if name == "FrameProcessor":
        from .processor import FrameProcessor
        return FrameProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
