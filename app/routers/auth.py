# app/routers/auth.py
"""
Auth API: login / logout / me / signup. The administrator account is
hardcoded via the ADMIN_EMAIL / ADMIN_PASSWORD environment variables and
provisioned on startup (seed_admin_from_env) — there is no first-run setup flow.
"""

import re
import secrets
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response

from .. import config
from ..config import SESSION_COOKIE, COOKIE_SECURE
from ..db import _DB_LOCK, db_connect, audit_log
from ..security import _verify_password, _pbkdf2_hash_password
from ..time_utils import utc_now_iso
from ..auth import get_current_user, _get_session_id_from_request

router = APIRouter()

_SESSION_DAYS = 7
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MIN_PASSWORD_LEN = 8


def _start_session(resp: Response, user_id: int) -> None:
    """Create a session row and set the session cookie on the response."""
    sid = secrets.token_urlsafe(32)
    now_iso = utc_now_iso()
    expires_at_iso = (datetime.now(timezone.utc) + timedelta(days=_SESSION_DAYS)).isoformat()

    with _DB_LOCK:
        conn = db_connect()
        conn.execute("""
            INSERT INTO sessions (id, user_id, created_at, last_seen_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
        """, (sid, user_id, now_iso, now_iso, expires_at_iso))
        conn.commit()
        conn.close()

    resp.set_cookie(
        key=SESSION_COOKIE,
        value=sid,
        httponly=True,
        samesite="lax",
        secure=COOKIE_SECURE,
        max_age=_SESSION_DAYS * 24 * 3600,
        path="/",
    )


def seed_admin_from_env() -> None:
    """
    Provision the hardcoded administrator from the ADMIN_EMAIL / ADMIN_PASSWORD
    environment variables. Called once on startup. The env vars are the source
    of truth: if the admin row is missing it is created; if it already exists
    its password is refreshed and the account is forced to role=admin/active.

    There is intentionally no first-run "first login becomes admin" flow.
    """
    email = (config.ADMIN_EMAIL or "").lower().strip()
    password = config.ADMIN_PASSWORD or ""
    if not email or not password:
        print(
            "[auth] WARNING: ADMIN_EMAIL / ADMIN_PASSWORD are not set — "
            "no administrator account was provisioned.",
            flush=True,
        )
        return

    pw = _pbkdf2_hash_password(password)
    with _DB_LOCK:
        conn = db_connect()
        row = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if row:
            conn.execute(
                """
                UPDATE users
                SET role = 'admin', is_active = 1,
                    pw_algo = ?, pw_iters = ?, pw_salt_b64 = ?, pw_hash_b64 = ?
                WHERE id = ?
                """,
                (pw["algo"], int(pw["iters"]), pw["salt_b64"], pw["hash_b64"], row["id"]),
            )
        else:
            conn.execute(
                """
                INSERT INTO users (email, display_name, role, pw_algo, pw_iters, pw_salt_b64, pw_hash_b64, created_at, is_active)
                VALUES (?, ?, 'admin', ?, ?, ?, ?, ?, 1)
                """,
                (email, "Admin", pw["algo"], int(pw["iters"]), pw["salt_b64"], pw["hash_b64"], utc_now_iso()),
            )
        conn.commit()
        conn.close()
    print(f"[auth] Administrator account ready: {email}", flush=True)


def _create_user(email: str, password: str, role: str, display_name: Optional[str]) -> int:
    """Insert a user, returning the new id. Raises 409 on duplicate email."""
    pw = _pbkdf2_hash_password(password)
    with _DB_LOCK:
        conn = db_connect()
        try:
            cur = conn.execute("""
                INSERT INTO users (email, display_name, role, pw_algo, pw_iters, pw_salt_b64, pw_hash_b64, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (email, display_name, role, pw["algo"], int(pw["iters"]), pw["salt_b64"], pw["hash_b64"], utc_now_iso()))
            conn.commit()
            new_id = int(cur.lastrowid)
        except sqlite3.IntegrityError:
            conn.close()
            raise HTTPException(status_code=409, detail="An account with that email already exists")
        conn.close()
    return new_id


def _validate_credentials(email: str, password: str) -> None:
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Please enter a valid email address")
    if len(password) < _MIN_PASSWORD_LEN:
        raise HTTPException(status_code=400, detail=f"Password must be at least {_MIN_PASSWORD_LEN} characters")


@router.post("/api/auth/login")
def login(resp: Response, payload: dict = Body(...)):
    email = str(payload.get("email", "")).lower().strip()
    password = str(payload.get("password", ""))

    if not email or not password:
        raise HTTPException(status_code=400, detail="Missing email or password")

    with _DB_LOCK:
        conn = db_connect()
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

    if not row:
        audit_log("login_failed", email, {"reason": "no_such_user"})
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user = dict(row)
    stored = {
        "algo": user["pw_algo"],
        "iters": str(user["pw_iters"]),
        "salt_b64": user["pw_salt_b64"],
        "hash_b64": user["pw_hash_b64"],
    }

    if int(user.get("is_active", 0)) != 1 or not _verify_password(password, stored):
        audit_log("login_failed", email, {"reason": "bad_password_or_inactive"})
        raise HTTPException(status_code=401, detail="Invalid credentials")

    _start_session(resp, user["id"])
    audit_log("login_ok", email, {"role": user["role"]})

    return {
        "ok": True,
        "user": {"email": user["email"], "display_name": user.get("display_name"), "role": user["role"]},
    }


@router.post("/api/auth/signup")
def signup(resp: Response, payload: dict = Body(...)):
    """
    Public self-registration. Role is always forced to 'staff' — the client
    role (if any) is ignored to prevent privilege escalation. The new account
    is active immediately and auto-logged-in.
    """
    email = str(payload.get("email", "")).lower().strip()
    password = str(payload.get("password", ""))
    display_name = str(payload.get("display_name", "")).strip() or None

    _validate_credentials(email, password)

    new_id = _create_user(email, password, "staff", display_name)
    _start_session(resp, new_id)
    audit_log("user_signup", email, {"role": "staff"})

    return {"ok": True, "user": {"email": email, "display_name": display_name, "role": "staff"}}


@router.post("/api/auth/logout")
def logout(resp: Response, user: dict = Depends(get_current_user), request: Request = None):
    sid = _get_session_id_from_request(request) if request is not None else None
    if sid:
        with _DB_LOCK:
            conn = db_connect()
            conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
            conn.commit()
            conn.close()

    resp.delete_cookie(SESSION_COOKIE, path="/")
    audit_log("logout", user.get("email"), {})
    return {"ok": True}


@router.get("/api/auth/me")
def me(user: dict = Depends(get_current_user)):
    return {"ok": True, "user": user}
