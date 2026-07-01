# app/main.py
"""
FastAPI application: wires routers, startup threads, and the static frontend.

Deployment role (WOODCHIP_ROLE) decides the live-data source:
- "device" (default): start the in-process inference loop (live_cam_trt) and the
  event sampler, exactly like the Jetson all-in-one build.
- "cloud": do NOT start in-process inference (no CUDA). Browsers stream frames to
  the ingest WebSocket, which calls the private inference service; the event
  sampler snapshots the per-device live store instead.

``live_cam_trt`` is imported lazily (only in the device branch) so the cloud
image never pulls the CUDA/TensorRT engine code.
"""

import os
import threading

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import config
from .db import db_init, close_pool
from .sampler import _event_sampler_loop
from .routers import (
    health,
    auth,
    admin,
    live,
    devices,
    rules,
    config as config_router,
    events,
    audit,
    ingest,
)


# ==========================================================
# App
# ==========================================================
app = FastAPI(title="Woodchip Backend (RBAC + Events + Rules + Audit)")

# Register routers
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(live.router)
app.include_router(devices.router)
app.include_router(rules.router)
app.include_router(config_router.router)
app.include_router(events.router)
app.include_router(audit.router)
app.include_router(ingest.router)


# ==========================================================
# Inference + sampler startup
# ==========================================================
@app.on_event("startup")
def on_startup():
    db_init()
    # The administrator is hardcoded via the ADMIN_EMAIL / ADMIN_PASSWORD
    # environment variables and provisioned here on every startup. There is no
    # first-run setup flow; the signup page is for regular (staff) users only.
    auth.seed_admin_from_env()

    # Start the in-process inference loop ONLY in the device role (CUDA stability).
    if config.ROLE == "device" and not config._inference_started:
        config._inference_started = True
        import live_cam_trt  # heavy path stays out of the cloud build
        t = threading.Thread(
            target=live_cam_trt.run_inference_loop,
            kwargs={"headless": True},
            daemon=True,
        )
        t.start()

    # Start event sampler once (writes events from whichever live source applies).
    if not config._sampler_started:
        config._sampler_started = True
        t2 = threading.Thread(target=_event_sampler_loop, daemon=True)
        t2.start()


@app.on_event("shutdown")
def on_shutdown():
    # Close the Postgres pool cleanly (no-op for SQLite).
    close_pool()


# ==========================================================
# Static frontend (built Vite app in web/). Mounted LAST so it never shadows the
# /api/* and /ping routes above; serves index.html for the SPA.
# ==========================================================
_WEB_DIR = os.path.join(config.BASE_DIR, "web")
if os.path.isdir(_WEB_DIR):
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")


# ==========================================================
# Optional CLI entry
# ==========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend_app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
    )
