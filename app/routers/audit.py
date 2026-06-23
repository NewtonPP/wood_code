# app/routers/audit.py
"""
Audit API.
"""

import json

from fastapi import APIRouter, Depends, Query

from ..db import _DB_LOCK, db_connect
from ..auth import require_perm

router = APIRouter()


@router.get("/api/audit")
def audit_list(
    user: dict = Depends(require_perm("view_audit")),
    limit: int = Query(200, ge=1, le=2000),
):
    with _DB_LOCK:
        conn = db_connect()
        rows = conn.execute("""
            SELECT id, ts, user_email, action, details_json
            FROM audit_log
            ORDER BY id DESC
            LIMIT ?
        """, (int(limit),)).fetchall()
        conn.close()

    out = []
    for r in rows:
        try:
            details = json.loads(r["details_json"]) if r["details_json"] else {}
        except Exception:
            details = {}
        out.append({
            "id": int(r["id"]),
            "ts": r["ts"],
            "user_email": r["user_email"],
            "action": r["action"],
            "details": details,
        })

    return {"ok": True, "items": out}
