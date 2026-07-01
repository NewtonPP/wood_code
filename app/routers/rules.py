# app/routers/rules.py
"""
Quality Rules: current + versioning + apply + rollback.
"""

import json
import sqlite3
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from .. import config
from ..db import _DB_LOCK, db_connect, audit_log
from ..time_utils import utc_now_iso
from ..auth import require_perm

router = APIRouter()


# Rule knobs that feed the inference runtime (kept in one place for both roles).
_RULE_KEYS = (
    "conf_thr", "nms_iou", "alarm_threshold_mm", "alarm_enabled", "ref_diam_mm",
    "histogram_mode", "moisture_enabled", "moisture_topk", "moisture_every_n_frames",
)


def _runtime_config() -> dict:
    """Current effective runtime config, role-aware."""
    if config.ROLE == "cloud":
        import woodchip_core
        return woodchip_core.get_runtime_config()
    import live_cam_trt
    return live_cam_trt.get_runtime_config() if hasattr(live_cam_trt, "get_runtime_config") else {}


def _apply_rules_to_runtime(rules: dict) -> None:
    """Push rule changes to the live inference runtime for the current role.

    - device: mutate live_cam_trt's in-process config (the loop reads it live).
    - cloud:  keep the backend's own config copy in sync AND forward to the
              private inference service (best-effort; rules are still persisted).
    """
    kwargs = {k: rules.get(k) for k in _RULE_KEYS}
    if config.ROLE == "cloud":
        import woodchip_core
        woodchip_core.update_runtime_config(**kwargs)
        if config.INFERENCE_URL:
            try:
                import httpx
                payload = {k: v for k, v in kwargs.items() if v is not None}
                httpx.post(f"{config.INFERENCE_URL}/config", json=payload, timeout=5.0)
            except Exception:
                pass  # rules are persisted regardless; propagation is best-effort
    else:
        import live_cam_trt
        if hasattr(live_cam_trt, "update_runtime_config"):
            live_cam_trt.update_runtime_config(**kwargs)


def _get_latest_rules_version() -> Optional[sqlite3.Row]:
    with _DB_LOCK:
        conn = db_connect()
        row = conn.execute("""
            SELECT * FROM quality_rules_versions
            ORDER BY version DESC
            LIMIT 1
        """).fetchone()
        conn.close()
        return row


def _insert_rules_version(rules: dict, created_by: Optional[str], reason: Optional[str], apply_now: bool) -> dict:
    last = _get_latest_rules_version()
    next_ver = int(last["version"]) + 1 if last else 1

    rec = {
        "version": next_ver,
        "created_at": utc_now_iso(),
        "created_by_email": created_by,
        "reason": reason,
        "rules_json": json.dumps(rules, ensure_ascii=False),
        "applied": 1 if apply_now else 0,
        "applied_at": utc_now_iso() if apply_now else None,
    }

    with _DB_LOCK:
        conn = db_connect()
        conn.execute("""
            INSERT INTO quality_rules_versions
            (version, created_at, created_by_email, reason, rules_json, applied, applied_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            rec["version"], rec["created_at"], rec["created_by_email"], rec["reason"],
            rec["rules_json"], rec["applied"], rec["applied_at"]
        ))
        conn.commit()
        conn.close()

    return rec


@router.get("/api/rules/current")
def rules_current(user: dict = Depends(require_perm("view_live"))):
    row = _get_latest_rules_version()
    if not row:
        # no saved versions yet -> return current runtime config if available
        cfg = _runtime_config()
        return {"ok": True, "source": "runtime", "version": None, "rules": cfg}
    return {
        "ok": True,
        "source": "db",
        "version": int(row["version"]),
        "created_at": row["created_at"],
        "created_by_email": row["created_by_email"],
        "reason": row["reason"],
        "applied": bool(row["applied"]),
        "applied_at": row["applied_at"],
        "rules": json.loads(row["rules_json"]),
    }


@router.get("/api/rules/versions")
def rules_versions(
    user: dict = Depends(require_perm("edit_rules")),
    limit: int = Query(50, ge=1, le=500),
):
    with _DB_LOCK:
        conn = db_connect()
        rows = conn.execute("""
            SELECT version, created_at, created_by_email, reason, applied, applied_at
            FROM quality_rules_versions
            ORDER BY version DESC
            LIMIT ?
        """, (int(limit),)).fetchall()
        conn.close()

    out = []
    for r in rows:
        out.append({
            "version": int(r["version"]),
            "created_at": r["created_at"],
            "created_by_email": r["created_by_email"],
            "reason": r["reason"],
            "applied": bool(r["applied"]),
            "applied_at": r["applied_at"],
        })
    return {"ok": True, "versions": out}


@router.post("/api/rules/update")
def rules_update(
    payload: dict = Body(...),
    user: dict = Depends(require_perm("edit_rules")),
):
    """
    Creates a new rules version and applies immediately (MVP).
    """
    rules = payload.get("rules")
    reason = payload.get("reason")

    if not isinstance(rules, dict):
        raise HTTPException(status_code=400, detail="payload.rules must be an object")

    # Apply to runtime config (keep contract compatible with your existing /api/config)
    _apply_rules_to_runtime(rules)

    rec = _insert_rules_version(rules=rules, created_by=user.get("email"), reason=reason, apply_now=True)
    audit_log("rules_updated", user.get("email"), {"version": rec["version"], "reason": reason, "rules": rules})
    return {"ok": True, "version": rec["version"], "applied": True, "rules": rules}


@router.post("/api/rules/rollback")
def rules_rollback(
    payload: dict = Body(...),
    user: dict = Depends(require_perm("edit_rules")),
):
    """
    Roll back by creating a NEW version equal to a previous version's rules, then applying.
    """
    target_version = int(payload.get("version", 0))
    if target_version <= 0:
        raise HTTPException(status_code=400, detail="payload.version must be > 0")

    with _DB_LOCK:
        conn = db_connect()
        row = conn.execute("SELECT * FROM quality_rules_versions WHERE version = ?", (target_version,)).fetchone()
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Version not found")

    rules = json.loads(row["rules_json"])

    _apply_rules_to_runtime(rules)

    rec = _insert_rules_version(rules=rules, created_by=user.get("email"), reason=f"rollback_to_v{target_version}", apply_now=True)
    audit_log("rules_rollback", user.get("email"), {"to_version": target_version, "new_version": rec["version"]})
    return {"ok": True, "rolled_back_to": target_version, "new_version": rec["version"], "rules": rules}
