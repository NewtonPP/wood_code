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
code lives in ``loop``/``engines`` and is only imported when the *real* loop
runs. On machines without TensorRT/pycuda (e.g. a dev Mac) a synthetic ``mock``
loop is used instead so the backend + frontend can be tested end-to-end.
"""

import os
import importlib.util

from .config import get_runtime_config, update_runtime_config
from .state import (
    get_health,
    get_latest_frame,
    get_latest_stats,
    get_hist_data,
    get_latest_moisture,
    get_latest_health,
)


def _use_mock() -> bool:
    """
    Decide whether to run the synthetic (mock) inference loop.

    - WOODCHIP_FAKE_INFERENCE=1/true/yes/on  -> force mock
    - WOODCHIP_FAKE_INFERENCE=0/false/no/off -> force real
    - unset -> auto: use mock when tensorrt/pycuda aren't installed.
    """
    v = os.environ.get("WOODCHIP_FAKE_INFERENCE")
    if v is not None:
        return v.strip().lower() in ("1", "true", "yes", "on")
    has_trt = importlib.util.find_spec("tensorrt") is not None
    has_cuda = importlib.util.find_spec("pycuda") is not None
    return not (has_trt and has_cuda)


def run_inference_loop(headless: bool = True):
    """Dispatch to the real TensorRT loop or the mock loop."""
    if _use_mock():
        print("[live_cam_trt] Using MOCK inference loop (no TensorRT/CUDA).", flush=True)
        from .mock import run_mock_inference_loop
        return run_mock_inference_loop(headless=headless)

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
