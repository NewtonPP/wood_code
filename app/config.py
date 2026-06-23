# app/config.py
"""
Paths, DB location, session cookie name, hardcoded admin credentials, and
background-thread flags.
"""

import os

try:
    # Load a local .env (if present) so ADMIN_EMAIL/ADMIN_PASSWORD and friends
    # work in development. In cloud deployments these are set as real
    # environment variables on the platform (AWS task/service config), so the
    # missing-.env case is a no-op.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


# Repo root (the directory that contains this package)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.environ.get("WOODCHIP_DB_PATH", os.path.join(BASE_DIR, "woodchip_app.sqlite3"))

# Session cookie name
SESSION_COOKIE = "woodchip_session"

# Mark the session cookie as Secure (HTTPS-only). Enable in production by
# setting WOODCHIP_COOKIE_SECURE=1. The app is served over HTTPS in the cloud
# deployment, so this should be 1 there; left off by default so local http dev works.
COOKIE_SECURE = os.environ.get("WOODCHIP_COOKIE_SECURE", "0") == "1"

# Identifier for this deployment/instance. Shown in the UI so operators know
# which environment they're viewing. Mirrors the value the sampler reads in
# app/sampler.py.
DEVICE_ID = os.environ.get("WOODCHIP_DEVICE_ID", "device-1")

# Hardcoded administrator credentials. The admin account is provisioned from
# these environment variables on startup (see seed_admin_from_env in
# app/routers/auth.py) — there is no first-run "first login becomes admin"
# setup flow. The signup page is for regular (staff) users only.
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

# Background threads flags
_inference_started = False
_sampler_started = False
