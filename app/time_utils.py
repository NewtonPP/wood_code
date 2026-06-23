# app/time_utils.py
"""
Time helpers (UTC ISO / epoch + safe ISO parsing across Python versions).
"""

import time
from datetime import datetime, timezone
from typing import Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_epoch() -> float:
    return time.time()


def _parse_iso_dt(s: str) -> Optional[datetime]:
    """
    Parse ISO datetime safely across Python versions (incl. Py3.6).
    Returns tz-aware UTC datetime when possible.
    """
    if not s:
        return None

    s2 = s.strip().replace("Z", "+00:00")

    # Python 3.7+ supports datetime.fromisoformat
    try:
        dt = datetime.fromisoformat(s2)  # may not exist on Py3.6
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass

    # Fallback: dateutil if available
    try:
        from dateutil.parser import isoparse  # type: ignore
        dt = isoparse(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass

    # Last resort: strip offset/micros; assume UTC
    try:
        base = s2
        if "+" in base:
            base = base.split("+", 1)[0]
        if "-" in base[19:]:  # handle negative offset like ...-05:00
            base = base[:19]
        if "." in base:
            base = base.split(".", 1)[0]
        dt = datetime.strptime(base, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None
