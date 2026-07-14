#!/usr/bin/env bash
# Local dev launcher — runs the same three-process stack as production:
#   :9000 inference service (real ONNX model)
#   :8000 backend (cloud role, WebSocket ingest)
#   :5173 frontend dev server (hot reload; proxies /api to :8000)
#
# Prerequisites (first run):
#   python3 -m venv .venv
#   .venv/bin/pip install -r requirements-cloud.txt -r inference_service/requirements.txt
#   inference_service/models/detr_resnet101.onnx  (export per DEPLOYMENT.md §5)
#   ADMIN_EMAIL / ADMIN_PASSWORD in .env (or exported) to seed the admin login
set -e
cd "$(dirname "$0")"

if [ ! -f inference_service/models/detr_resnet101.onnx ] && [ -z "$DETR_ONNX_PATH" ]; then
  echo "ERROR: DETR model not found at inference_service/models/detr_resnet101.onnx."
  echo "Export it first (DEPLOYMENT.md, 'Model files') or set DETR_ONNX_PATH."
  exit 1
fi

# Install frontend deps if missing (first run only)
if [ ! -d frontend/node_modules ]; then
  echo "Installing frontend dependencies..."
  ( cd frontend && npm install )
fi

echo "Starting inference -> http://localhost:9000"
.venv/bin/uvicorn inference_service.main:app --host 127.0.0.1 --port 9000 &
INFER_PID=$!

echo "Starting backend   -> http://localhost:8000"
WOODCHIP_ROLE=cloud INFERENCE_URL=http://localhost:9000 \
  .venv/bin/uvicorn backend_app:app --host 0.0.0.0 --port 8000 --workers 1 &
BACKEND_PID=$!

# When this script exits (or you press Ctrl+C), stop both servers too
trap 'echo; echo "Shutting down..."; kill "$BACKEND_PID" "$INFER_PID" 2>/dev/null' EXIT INT TERM

# Start the frontend (Vite dev) on :5173 and open it in the browser.
echo "Starting frontend  -> http://localhost:5173"
cd frontend
npm run dev -- --open
