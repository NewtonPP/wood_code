# live_cam_trt/__init__.py
"""
Unified drop-in: DETR (size + histogram + calibration + alarm) + Moisture (TRT).

Public API consumed by the backend:
- run_inference_loop(headless=True)
- get_runtime_config() / update_runtime_config(...)
- get_health()
- shared-state accessors: get_latest_frame/stats, get_hist_data,
  get_latest_moisture, get_latest_health

The package is split into focused modules (config, state, engines, geometry,
reference, moisture, overlay, loop) but behaves identically to the original
single-file module.

Importing this package is intentionally lightweight: the CUDA/TensorRT engine
code lives in ``loop``/``engines`` and is only imported when the loop actually
starts, so the backend can import this package for config/state access without
pulling in CUDA. Running the loop requires TensorRT/pycuda (Jetson/JetPack).
"""

from .config import get_runtime_config, update_runtime_config
from .state import (
    get_health,
    get_latest_frame,
    get_latest_stats,
    get_hist_data,
    get_latest_moisture,
    get_latest_health,
)


def run_inference_loop(headless: bool = True):
    """Run the TensorRT inference loop (requires tensorrt/pycuda — Jetson)."""
    from .loop import run_inference_loop as _real_loop  # heavy: imports tensorrt/pycuda
    return _real_loop(headless=headless)


def main():
    run_inference_loop(headless=False)


__all__ = [
    "run_inference_loop",
    "main",
    "get_runtime_config",
    "update_runtime_config",
    "get_health",
    "get_latest_frame",
    "get_latest_stats",
    "get_hist_data",
    "get_latest_moisture",
    "get_latest_health",
]
