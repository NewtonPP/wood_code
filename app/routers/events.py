# app/routers/events.py
"""
Events API: list + export CSV.
"""

from typing import Optional, List, Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from ..db import _DB_LOCK, db_connect, audit_log
from ..auth import require_perm
from .. import data_source

router = APIRouter()

# Roles allowed to see ALL tenants' events (and to filter by an arbitrary
# device_id). Everyone else is hard-scoped to their own device — passing a
# foreign device_id has no effect, so a tenant can never read another's events.
OVERSIGHT_ROLES = {"admin", "manager"}


def _effective_device_filter(user: dict, device_id_param: Optional[str]) -> Optional[str]:
    if user.get("role") in OVERSIGHT_ROLES:
        return device_id_param  # None = all devices; or a chosen device
    return data_source.device_id_for_user(user)  # forced to the caller's own device


@router.get("/api/events")
def events_list(
    user: dict = Depends(require_perm("view_events")),
    start_epoch: Optional[float] = Query(None, description="UTC epoch seconds"),
    end_epoch: Optional[float] = Query(None, description="UTC epoch seconds"),
    device_id: Optional[str] = Query(None),
    alarm_only: bool = Query(False),
    limit: int = Query(200, ge=1, le=5000),
):
    q = "SELECT * FROM events WHERE 1=1"
    params: List[Any] = []

    if start_epoch is not None:
        q += " AND ts_epoch >= ?"
        params.append(float(start_epoch))
    if end_epoch is not None:
        q += " AND ts_epoch <= ?"
        params.append(float(end_epoch))
    eff_device = _effective_device_filter(user, device_id)
    if eff_device:
        q += " AND device_id = ?"
        params.append(eff_device)
    if alarm_only:
        q += " AND alarm_active = 1"

    q += " ORDER BY ts_epoch DESC LIMIT ?"
    params.append(int(limit))

    with _DB_LOCK:
        conn = db_connect()
        rows = conn.execute(q, tuple(params)).fetchall()
        conn.close()

    out = []
    for r in rows:
        out.append({
            "id": int(r["id"]),
            "ts": r["ts"],
            "ts_epoch": float(r["ts_epoch"]),
            "device_id": r["device_id"],
            "alarm_active": bool(r["alarm_active"]),
            "alarm_max_d_mm": r["alarm_max_d_mm"],
            "mean_d": r["mean_d"],
            "std_d": r["std_d"],
            "units": r["units"],
            "moisture_mean_pred": r["moisture_mean_pred"],
        })

    return {"ok": True, "events": out}


@router.get("/api/events/export.csv")
def events_export_csv(
    user: dict = Depends(require_perm("export_events")),
    start_epoch: Optional[float] = Query(None),
    end_epoch: Optional[float] = Query(None),
    device_id: Optional[str] = Query(None),
    alarm_only: bool = Query(False),
    limit: int = Query(10000, ge=1, le=200000),
):
    q = "SELECT * FROM events WHERE 1=1"
    params: List[Any] = []

    if start_epoch is not None:
        q += " AND ts_epoch >= ?"
        params.append(float(start_epoch))
    if end_epoch is not None:
        q += " AND ts_epoch <= ?"
        params.append(float(end_epoch))
    eff_device = _effective_device_filter(user, device_id)
    if eff_device:
        q += " AND device_id = ?"
        params.append(eff_device)
    if alarm_only:
        q += " AND alarm_active = 1"

    q += " ORDER BY ts_epoch ASC LIMIT ?"
    params.append(int(limit))

    with _DB_LOCK:
        conn = db_connect()
        rows = conn.execute(q, tuple(params)).fetchall()
        conn.close()

    audit_log("events_export_csv", user.get("email"), {
        "start_epoch": start_epoch, "end_epoch": end_epoch, "device_id": device_id,
        "alarm_only": alarm_only, "limit": limit, "rows": len(rows)
    })

    def gen():
        header = [
            "ts", "ts_epoch", "device_id",
            "alarm_active", "alarm_max_d_mm",
            "mean_d", "std_d", "units",
            "moisture_mean_pred",
        ]
        yield (",".join(header) + "\n").encode("utf-8")

        for r in rows:
            line = [
                str(r["ts"] or ""),
                str(r["ts_epoch"] or ""),
                str(r["device_id"] or ""),
                str(int(r["alarm_active"] or 0)),
                str(r["alarm_max_d_mm"] if r["alarm_max_d_mm"] is not None else ""),
                str(r["mean_d"] if r["mean_d"] is not None else ""),
                str(r["std_d"] if r["std_d"] is not None else ""),
                str(r["units"] or ""),
                str(r["moisture_mean_pred"] or ""),
            ]
            yield (",".join(line) + "\n").encode("utf-8")

    return StreamingResponse(
        gen(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="events_export.csv"'},
    )
