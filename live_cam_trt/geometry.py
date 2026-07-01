# live_cam_trt/geometry.py
"""
Compatibility shim — the implementations now live in ``woodchip_core.geometry``
so the Jetson device and the cloud share one CUDA-free post-processing path.

This module preserves the original signatures (``compute_LWD`` reading the
module-global calibration ``config.PIXELS_PER_MM``) so ``loop.py`` is unchanged.
"""

from woodchip_core.geometry import (  # noqa: F401
    _now,
    softmax_lastdim,
    safe_softmax_2d,
    nms_np,
    resize_keep_aspect,
    preprocess_fixed,
    postprocess_from_fixed,
    compute_LWD as _compute_LWD,
)

from . import config


def compute_LWD(boxes_xyxy):
    """Device wrapper: use the single-camera module-global calibration."""
    return _compute_LWD(boxes_xyxy, config.PIXELS_PER_MM)
