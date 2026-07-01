# app/routers/admin.py
"""
Admin: manage users (list / create / update). Requires the `manage_users`
permission, which only the admin role has.
"""

from fastapi import APIRouter, Body, Depends, HTTPException

from ..db import _DB_LOCK, db_connect, audit_log, IntegrityError
from ..security import _pbkdf2_hash_password
from ..time_utils import utc_now_iso
from ..rbac import ROLE_PERMS
from ..auth import require_perm

router = APIRouter()


@router.get("/api/admin/users")
def admin_list_users(admin: dict = Depends(require_perm("manage_users"))):
    with _DB_LOCK:
        conn = db_connect()
        rows = conn.execute("""
            SELECT id, email, display_name, role, is_active, created_at
            FROM users
            ORDER BY created_at ASC
        """).fetchall()
        conn.close()
    return {"ok": True, "users": [dict(r) for r in rows]}


@router.post("/api/admin/users")
def admin_create_user(
    payload: dict = Body(...),
    admin: dict = Depends(require_perm("manage_users")),
):
    email = str(payload.get("email", "")).lower().strip()
    password = str(payload.get("password", "")).strip()
    role = str(payload.get("role", "")).strip() or "staff"
    display_name = str(payload.get("display_name", "")).strip() or None

    if role not in ROLE_PERMS:
        raise HTTPException(status_code=400, detail=f"Unknown role: {role}")
    if not email or not password:
        raise HTTPException(status_code=400, detail="Missing email or password")

    pw = _pbkdf2_hash_password(password)

    with _DB_LOCK:
        conn = db_connect()
        try:
            conn.execute("""
                INSERT INTO users (email, display_name, role, pw_algo, pw_iters, pw_salt_b64, pw_hash_b64, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (email, display_name, role, pw["algo"], int(pw["iters"]), pw["salt_b64"], pw["hash_b64"], utc_now_iso()))
            conn.commit()
        except IntegrityError:
            conn.close()
            raise HTTPException(status_code=409, detail="User already exists")
        conn.close()

    audit_log("user_created", admin.get("email"), {"email": email, "role": role})
    return {"ok": True, "email": email, "role": role}


@router.post("/api/admin/users/{user_id}")
def admin_update_user(
    user_id: int,
    payload: dict = Body(...),
    admin: dict = Depends(require_perm("manage_users")),
):
    """Update a user's role and/or active status (e.g. promote to admin)."""
    sets = []
    params = []

    if "role" in payload:
        role = str(payload.get("role", "")).strip()
        if role not in ROLE_PERMS:
            raise HTTPException(status_code=400, detail=f"Unknown role: {role}")
        sets.append("role = ?")
        params.append(role)

    if "is_active" in payload:
        is_active = 1 if payload.get("is_active") else 0
        # Don't let an admin deactivate their own account and lock themselves out.
        if is_active == 0 and admin.get("id") == user_id:
            raise HTTPException(status_code=400, detail="You cannot deactivate your own account")
        sets.append("is_active = ?")
        params.append(is_active)

    if not sets:
        raise HTTPException(status_code=400, detail="Nothing to update")

    params.append(user_id)
    with _DB_LOCK:
        conn = db_connect()
        cur = conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE id = ?", params)
        conn.commit()
        changed = cur.rowcount
        conn.close()

    if not changed:
        raise HTTPException(status_code=404, detail="User not found")

    audit_log("user_updated", admin.get("email"), {"user_id": user_id, "changes": payload})
    return {"ok": True}
