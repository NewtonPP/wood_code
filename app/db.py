# app/db.py
"""
Database layer: SQLite (local/dev, Jetson) or PostgreSQL (cloud).

Backend is chosen by the ``DATABASE_URL`` env var:
- set to ``postgresql://…`` (incl. Cloud SQL ``…?host=/cloudsql/INSTANCE``) -> Postgres
- unset -> SQLite at ``DB_PATH`` (unchanged behaviour)

The Postgres path wraps the connection so the existing routers — which call
``conn.execute("… ?", params).fetchone()/fetchall()`` and ``dict(row)`` — keep
working without edits: the wrapper translates ``?`` placeholders to ``%s`` and
psycopg's ``dict_row`` rows are already dict-like. ``db_connect()`` /
``db_init()`` / ``audit_log()`` keep their names and call signatures.

Concurrency: SQLite is guarded by the process-wide ``_DB_LOCK``; Postgres uses a
connection pool and ``_DB_LOCK`` becomes a no-op context so DB access is not
serialized (the existing ``with _DB_LOCK:`` blocks are unaffected either way).
"""

import os
import json
import sqlite3
import threading
from contextlib import nullcontext
from typing import Optional

from .config import DB_PATH
from .time_utils import utc_now_iso


DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
IS_POSTGRES = DATABASE_URL.startswith("postgres")


if IS_POSTGRES:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    # Exception callers catch on duplicate-key inserts (see auth/admin routers).
    IntegrityError = psycopg.errors.UniqueViolation
    # Pool is thread-safe; no global serialization needed.
    _DB_LOCK = nullcontext()

    _POOL: Optional["ConnectionPool"] = None

    def _pool() -> "ConnectionPool":
        global _POOL
        if _POOL is None:
            # autocommit: this is a request/response app doing mostly single-
            # statement writes; it keeps pooled connections transaction-clean
            # (no rollback churn on return). Explicit conn.commit() calls become
            # harmless no-ops.
            _POOL = ConnectionPool(
                DATABASE_URL,
                kwargs={"row_factory": dict_row, "autocommit": True},
                min_size=1,
                max_size=10,
                open=True,
            )
        return _POOL

    def close_pool() -> None:
        """Close the connection pool on app shutdown (frees the bg worker thread)."""
        global _POOL
        if _POOL is not None:
            _POOL.close()
            _POOL = None

    class _PgConn:
        """Adapts a pooled psycopg connection to the sqlite3.Connection API the
        routers use (``?`` placeholders, ``.execute().fetchone()``, ``.commit()``,
        ``.close()`` returns the connection to the pool)."""

        def __init__(self, raw, pool):
            self._raw = raw
            self._pool = pool
            self._closed = False

        def execute(self, sql, params=()):
            return self._raw.execute(sql.replace("?", "%s"), params)

        def cursor(self):
            return self._raw.cursor()

        def commit(self):
            self._raw.commit()

        def close(self):
            if not self._closed:
                self._closed = True
                # putconn resets (rolls back) any open/aborted transaction.
                self._pool.putconn(self._raw)

else:
    IntegrityError = sqlite3.IntegrityError
    _DB_LOCK = threading.Lock()

    def close_pool() -> None:
        """No-op for SQLite (no pool); present so callers can call unconditionally."""
        return None


def db_connect():
    """Return a connection exposing the sqlite3-style API used across the app."""
    if IS_POSTGRES:
        return _PgConn(_pool().getconn(), _pool())
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def insert_returning_id(conn, sql, params):
    """Run an INSERT and return the new row's ``id`` on either backend.

    Replaces SQLite-only ``cursor.lastrowid`` (Postgres uses ``RETURNING id``).
    Does NOT commit — the caller commits as before.
    """
    if IS_POSTGRES:
        cur = conn.execute(sql + " RETURNING id", params)
        return int(cur.fetchone()["id"])
    cur = conn.execute(sql, params)
    return int(cur.lastrowid)


def _schema_statements():
    """DDL for both backends (idempotent). Differs only in id/float types."""
    if IS_POSTGRES:
        pk = "BIGSERIAL PRIMARY KEY"
        flt = "DOUBLE PRECISION"
    else:
        pk = "INTEGER PRIMARY KEY AUTOINCREMENT"
        flt = "REAL"

    return [
        f"""
        CREATE TABLE IF NOT EXISTS users (
            id {pk},
            email TEXT UNIQUE NOT NULL,
            display_name TEXT,
            role TEXT NOT NULL,
            pw_algo TEXT NOT NULL,
            pw_iters INTEGER NOT NULL,
            pw_salt_b64 TEXT NOT NULL,
            pw_hash_b64 TEXT NOT NULL,
            created_at TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS audit_log (
            id {pk},
            ts TEXT NOT NULL,
            user_email TEXT,
            action TEXT NOT NULL,
            details_json TEXT
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS quality_rules_versions (
            id {pk},
            version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            created_by_email TEXT,
            reason TEXT,
            rules_json TEXT NOT NULL,
            applied INTEGER NOT NULL DEFAULT 0,
            applied_at TEXT
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS events (
            id {pk},
            ts TEXT NOT NULL,
            ts_epoch {flt} NOT NULL,
            device_id TEXT,
            alarm_active INTEGER,
            alarm_max_d_mm {flt},
            mean_d {flt},
            std_d {flt},
            units TEXT,
            moisture_mean_pred TEXT,
            payload_json TEXT
        );
        """,
        # Device registry (device <-> owner) for tenant scoping + admin inventory.
        """
        CREATE TABLE IF NOT EXISTS devices (
            device_id TEXT PRIMARY KEY,
            owner_user_id INTEGER,
            label TEXT,
            created_at TEXT NOT NULL,
            last_seen_at TEXT
        );
        """,
        "CREATE INDEX IF NOT EXISTS idx_events_ts_epoch ON events(ts_epoch);",
        "CREATE INDEX IF NOT EXISTS idx_events_device ON events(device_id);",
    ]


def db_init() -> None:
    if not IS_POSTGRES:
        d = os.path.dirname(DB_PATH)
        if d:
            os.makedirs(d, exist_ok=True)

    with _DB_LOCK:
        conn = db_connect()
        for stmt in _schema_statements():
            conn.execute(stmt)
        conn.commit()
        conn.close()


def upsert_device(device_id: str, owner_user_id: Optional[int], label: Optional[str] = None) -> None:
    """Register/refresh a device <-> owner mapping (called when a stream connects).

    The UPSERT syntax (``ON CONFLICT(device_id) DO UPDATE``) is supported by both
    SQLite (3.24+) and Postgres.
    """
    now = utc_now_iso()
    with _DB_LOCK:
        conn = db_connect()
        conn.execute(
            """
            INSERT INTO devices (device_id, owner_user_id, label, created_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(device_id) DO UPDATE SET
                owner_user_id = excluded.owner_user_id,
                label = COALESCE(excluded.label, devices.label),
                last_seen_at = excluded.last_seen_at
            """,
            (device_id, owner_user_id, label, now, now),
        )
        conn.commit()
        conn.close()


def list_devices(owner_user_id: Optional[int] = None) -> list:
    """List devices. ``owner_user_id=None`` returns all (admin/oversight view)."""
    with _DB_LOCK:
        conn = db_connect()
        if owner_user_id is None:
            rows = conn.execute(
                "SELECT device_id, owner_user_id, label, created_at, last_seen_at "
                "FROM devices ORDER BY created_at ASC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT device_id, owner_user_id, label, created_at, last_seen_at "
                "FROM devices WHERE owner_user_id = ? ORDER BY created_at ASC",
                (owner_user_id,),
            ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def audit_log(action: str, user_email: Optional[str] = None, details: Optional[dict] = None) -> None:
    rec = {
        "ts": utc_now_iso(),
        "user_email": user_email,
        "action": action,
        "details_json": json.dumps(details or {}, ensure_ascii=False),
    }
    with _DB_LOCK:
        conn = db_connect()
        conn.execute(
            "INSERT INTO audit_log (ts, user_email, action, details_json) VALUES (?, ?, ?, ?)",
            (rec["ts"], rec["user_email"], rec["action"], rec["details_json"]),
        )
        conn.commit()
        conn.close()
