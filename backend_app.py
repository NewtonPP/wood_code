# backend_app.py
"""
Compatibility entrypoint.

The backend was refactored into the ``app`` package (see app/main.py). This shim
preserves the existing ``uvicorn backend_app:app`` entrypoint and any service
units that reference it.
"""

from app.main import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend_app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # IMPORTANT for Jetson
    )
