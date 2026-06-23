#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Install frontend deps if missing (first run only)
if [ ! -d frontend/node_modules ]; then
  echo "Installing frontend dependencies..."
  ( cd frontend && npm install )
fi

# Start the backend (FastAPI) on :8000 in the background
echo "Starting backend  -> http://localhost:8000"
.venv/bin/uvicorn backend_app:app --host 0.0.0.0 --port 8000 --workers 1 &
BACKEND_PID=$!

# When this script exits (or you press Ctrl+C), stop the backend too
trap 'echo; echo "Shutting down..."; kill "$BACKEND_PID" 2>/dev/null' EXIT INT TERM

# Start the frontend (Vite dev) on :5173 and open it in the browser.
# Vite proxies /api and /ping to the backend on :8000 (see frontend/vite.config.ts),
# so the UI talks to FastAPI without rendering anything from port 8000.
echo "Starting frontend -> http://localhost:5173"
cd frontend
npm run dev -- --open
