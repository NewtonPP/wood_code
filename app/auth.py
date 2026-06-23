# app/auth.py
"""
Session/auth helpers and FastAPI dependencies (get_current_user, require_perm).
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request

from .config import SESSION_COOKIE
from .db import _DB_LOCK, db_connect
from .time_utils import utc_now_iso, _parse_iso_dt
from .rbac import has_perm


def _get_session_id_from_request(request: Request) -> Optional[str]:
    return request.cookies.get(SESSION_COOKIE)


def _get_user_by_session(session_id: str) -> Optional[dict]:
    with _DB_LOCK:
        conn = db_connect()
        row = conn.execute("""
            SELECT s.id as session_id, s.expires_at, u.*
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.id = ?
        """, (session_id,)).fetchone()
        if not row:
            conn.close()
            return None

        # Expiry check (Python 3.6-safe)
        expires_at = row["expires_at"]
        exp_dt = _parse_iso_dt(expires_at)
        if exp_dt is None:
            conn.close()
            return None

        if exp_dt < datetime.now(timezone.utc):
            # delete expired
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            conn.close()
            return None

        # Update last_seen
        conn.execute("UPDATE sessions SET last_seen_at = ? WHERE id = ?", (utc_now_iso(), session_id))
        conn.commit()
        conn.close()

        return dict(row)


def get_current_user(request: Request) -> dict:
    sid = _get_session_id_from_request(request)
    if not sid:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = _get_user_by_session(sid)
    if not user or int(user.get("is_active", 0)) != 1:
        raise HTTPException(status_code=401, detail="Invalid session")

    return {
        "id": user["id"],
        "email": user["email"],
        "display_name": user.get("display_name"),
        "role": user["role"],
    }


def require_perm(perm: str):
    def _dep(user: dict = Depends(get_current_user)) -> dict:
        role = user.get("role", "")
        if not has_perm(role, perm):
            raise HTTPException(status_code=403, detail=f"Missing permission: {perm}")
        return user
    return _dep
