# app/routers/config.py
"""
Legacy runtime configuration endpoints (kept for compatibility).
Now protected by edit_rules; changes are also versioned via quality rules.

Role-aware via the shared helpers in app.routers.rules (so a cloud deployment
forwards changes to the inference service, same as /api/rules/update).
"""

from fastapi import APIRouter, Body, Depends

from ..db import audit_log
from ..auth import require_perm
from .rules import _insert_rules_version, _apply_rules_to_runtime, _runtime_config

router = APIRouter()


@router.get("/api/config")
def get_config(user: dict = Depends(require_perm("edit_rules"))):
    return _runtime_config()


@router.post("/api/config")
def set_config(cfg: dict = Body(...), user: dict = Depends(require_perm("edit_rules"))):
    """
    Backward-compatible config API:
    - applies to runtime immediately
    - ALSO creates a rules version entry (so changes are versioned)
    """
    _apply_rules_to_runtime(cfg)

    # Version it
    rec = _insert_rules_version(rules=cfg, created_by=user.get("email"), reason="legacy_api_config_update", apply_now=True)
    audit_log("config_updated", user.get("email"), {"version": rec["version"], "cfg": cfg})

    return _runtime_config()
