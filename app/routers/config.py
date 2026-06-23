# app/routers/config.py
"""
Legacy runtime configuration endpoints (kept for compatibility).
Now protected by edit_rules; changes are also versioned via quality rules.
"""

from fastapi import APIRouter, Body, Depends

import live_cam_trt  # owns CUDA + TensorRT

from ..db import audit_log
from ..auth import require_perm
from .rules import _insert_rules_version

router = APIRouter()


@router.get("/api/config")
def get_config(user: dict = Depends(require_perm("edit_rules"))):
    if hasattr(live_cam_trt, "get_runtime_config"):
        return live_cam_trt.get_runtime_config()
    return {}


@router.post("/api/config")
def set_config(cfg: dict = Body(...), user: dict = Depends(require_perm("edit_rules"))):
    """
    Backward-compatible config API:
    - applies to runtime immediately
    - ALSO creates a rules version entry (so changes are versioned)
    """
    if hasattr(live_cam_trt, "update_runtime_config"):
        live_cam_trt.update_runtime_config(
            conf_thr=cfg.get("conf_thr"),
            nms_iou=cfg.get("nms_iou"),
            alarm_threshold_mm=cfg.get("alarm_threshold_mm"),
            alarm_enabled=cfg.get("alarm_enabled"),
            ref_diam_mm=cfg.get("ref_diam_mm"),
            histogram_mode=cfg.get("histogram_mode"),
            moisture_enabled=cfg.get("moisture_enabled"),
            moisture_topk=cfg.get("moisture_topk"),
            moisture_every_n_frames=cfg.get("moisture_every_n_frames"),
        )

    # Version it
    rec = _insert_rules_version(rules=cfg, created_by=user.get("email"), reason="legacy_api_config_update", apply_now=True)
    audit_log("config_updated", user.get("email"), {"version": rec["version"], "cfg": cfg})

    return live_cam_trt.get_runtime_config() if hasattr(live_cam_trt, "get_runtime_config") else {}
