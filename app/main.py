# app/main.py
"""
FastAPI application: wires routers, startup threads (inference + sampler), and
the static frontend mount.
"""

import threading

from fastapi import FastAPI

import live_cam_trt  # owns CUDA + TensorRT

from . import config
from .db import db_init
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
)


# ==========================================================
# App
# ==========================================================
app = FastAPI(title="Woodchip Device Backend (RBAC + Events + Rules + Audit)")

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

    # Start inference loop once (critical for CUDA stability)
    if not config._inference_started:
        config._inference_started = True
        t = threading.Thread(
            target=live_cam_trt.run_inference_loop,
            kwargs={"headless": True},
            daemon=True,
        )
        t.start()

    # Start event sampler once (writes 1 row/sec into events table)
    if not config._sampler_started:
        config._sampler_started = True
        t2 = threading.Thread(target=_event_sampler_loop, daemon=True)
        t2.start()


# ==========================================================
# Optional CLI entry
# ==========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend_app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # IMPORTANT for Jetson
    )
