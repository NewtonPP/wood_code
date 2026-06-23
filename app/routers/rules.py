# app/routers/rules.py
"""
Quality Rules: current + versioning + apply + rollback.
"""

import json
import sqlite3
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

import live_cam_trt  # owns CUDA + TensorRT

from ..db import _DB_LOCK, db_connect, audit_log
from ..time_utils import utc_now_iso
from ..auth import require_perm

router = APIRouter()


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
        # no saved versions yet -> return current live_cam config if available
        cfg = live_cam_trt.get_runtime_config() if hasattr(live_cam_trt, "get_runtime_config") else {}
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
    if hasattr(live_cam_trt, "update_runtime_config"):
        live_cam_trt.update_runtime_config(
            conf_thr=rules.get("conf_thr"),
            nms_iou=rules.get("nms_iou"),
            alarm_threshold_mm=rules.get("alarm_threshold_mm"),
            alarm_enabled=rules.get("alarm_enabled"),
            ref_diam_mm=rules.get("ref_diam_mm"),
            histogram_mode=rules.get("histogram_mode"),
            moisture_enabled=rules.get("moisture_enabled"),
            moisture_topk=rules.get("moisture_topk"),
            moisture_every_n_frames=rules.get("moisture_every_n_frames"),
        )

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

    if hasattr(live_cam_trt, "update_runtime_config"):
        live_cam_trt.update_runtime_config(
            conf_thr=rules.get("conf_thr"),
            nms_iou=rules.get("nms_iou"),
            alarm_threshold_mm=rules.get("alarm_threshold_mm"),
            alarm_enabled=rules.get("alarm_enabled"),
            ref_diam_mm=rules.get("ref_diam_mm"),
            histogram_mode=rules.get("histogram_mode"),
            moisture_enabled=rules.get("moisture_enabled"),
            moisture_topk=rules.get("moisture_topk"),
            moisture_every_n_frames=rules.get("moisture_every_n_frames"),
        )

    rec = _insert_rules_version(rules=rules, created_by=user.get("email"), reason=f"rollback_to_v{target_version}", apply_now=True)
    audit_log("rules_rollback", user.get("email"), {"to_version": target_version, "new_version": rec["version"]})
    return {"ok": True, "rolled_back_to": target_version, "new_version": rec["version"], "rules": rules}
