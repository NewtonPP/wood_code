# app/db.py
"""
SQLite helpers: connection, schema init, and audit logging.
"""

import os
import json
import sqlite3
import threading
from typing import Optional

from .config import DB_PATH
from .time_utils import utc_now_iso


_DB_LOCK = threading.Lock()


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_init() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True) if os.path.dirname(DB_PATH) else None

    with _DB_LOCK:
        conn = db_connect()
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,               -- random token
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            user_email TEXT,
            action TEXT NOT NULL,
            details_json TEXT
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS quality_rules_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            created_by_email TEXT,
            reason TEXT,
            rules_json TEXT NOT NULL,
            applied INTEGER NOT NULL DEFAULT 0,
            applied_at TEXT
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,              -- ISO UTC
            ts_epoch REAL NOT NULL,
            device_id TEXT,
            alarm_active INTEGER,
            alarm_max_d_mm REAL,
            mean_d REAL,
            std_d REAL,
            units TEXT,
            moisture_mean_pred TEXT,
            payload_json TEXT              -- full snapshot for future-proofing
        );
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_ts_epoch ON events(ts_epoch);
        """)

        conn.commit()
        conn.close()


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
