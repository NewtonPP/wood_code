# app/routers/ingest.py
"""
Frame ingest WebSocket (cloud role).

The browser captures its USB camera with getUserMedia, downsizes frames to JPEG
and streams them here. For each frame we:
  1. forward it to the private inference service (INFERENCE_URL/infer),
  2. store the result in the per-device live store (so the polled
     /api/stats|hist|moisture endpoints serve it), and
  3. echo the JSON result back over the same socket so the browser can draw the
     AI overlay on its own local <video> immediately.

Auth: the session cookie is validated on connect and the user must have the
``view_live`` permission. Each connection is bound to that user's own device id
(tenant isolation) — a browser cannot push to or read another account's stream.
"""

import base64

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import config
from .. import live_store
from ..auth import _get_user_by_session
from ..config import SESSION_COOKIE
from ..data_source import device_id_for_user
from ..db import upsert_device
from ..rbac import has_perm

router = APIRouter()

# Reused across connections; created lazily so the device build (no httpx) is fine.
_http_client = None


def _client():
    global _http_client
    if _http_client is None:
        import httpx
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
    return _http_client


def _authenticate(ws: WebSocket):
    sid = ws.cookies.get(SESSION_COOKIE)
    if not sid:
        return None
    user = _get_user_by_session(sid)
    if not user or int(user.get("is_active", 0)) != 1:
        return None
    if not has_perm(user.get("role", ""), "view_live"):
        return None
    return {"id": user["id"], "email": user["email"], "role": user["role"]}


@router.websocket("/api/ingest/ws")
async def ingest_ws(ws: WebSocket):
    user = _authenticate(ws)
    if user is None:
        await ws.close(code=1008)  # policy violation
        return
    if config.ROLE != "cloud" or not config.INFERENCE_URL:
        await ws.accept()
        await ws.send_json({"ok": False, "error": "ingest disabled (not cloud role / INFERENCE_URL unset)"})
        await ws.close(code=1011)
        return

    device_id = device_id_for_user(user)
    # Register/refresh the device <-> owner mapping for the registry + tenant scoping.
    try:
        upsert_device(device_id, user["id"])
    except Exception:
        pass  # registry is best-effort; never block the stream
    await ws.accept()

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            # Accept either binary JPEG bytes or a base64 text frame.
            if msg.get("bytes") is not None:
                jpeg_b64 = base64.b64encode(msg["bytes"]).decode()
            elif msg.get("text") is not None:
                jpeg_b64 = msg["text"]
            else:
                continue

            try:
                resp = await _client().post(
                    f"{config.INFERENCE_URL}/infer",
                    json={"device_id": device_id, "frame_jpeg_b64": jpeg_b64},
                )
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                await ws.send_json({"ok": False, "error": f"inference unavailable: {e}"})
                continue

            live_store.update(
                device_id,
                stats=result.get("stats"),
                hist=result.get("histogram"),
                moisture=result.get("moisture"),
                health=result.get("health"),
            )
            await ws.send_json({"ok": True, "result": result})
    except WebSocketDisconnect:
        pass
