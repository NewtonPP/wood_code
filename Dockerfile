# Backend + built frontend image (cloud role). No CUDA/TensorRT/OpenCV.
# Build:  docker build -t woodchip-backend .
# Run:    docker run -p 8000:8000 \
#           -e WOODCHIP_ROLE=cloud -e INFERENCE_URL=http://inference:9000 \
#           -e ADMIN_EMAIL=... -e ADMIN_PASSWORD=... -e WOODCHIP_COOKIE_SECURE=1 \
#           woodchip-backend

# ---- Stage 1: build the Vite frontend into web/ ----
FROM node:20-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN cd frontend && npm ci
COPY frontend ./frontend
# vite.config.ts builds to ../web
RUN cd frontend && npm run build

# ---- Stage 2: Python backend ----
FROM python:3.11-slim AS backend
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WOODCHIP_ROLE=cloud

COPY requirements-cloud.txt ./
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Backend + shared post-processing config (no cv2 pulled at import).
COPY app ./app
COPY woodchip_core ./woodchip_core
COPY backend_app.py ./
# Built static frontend from stage 1.
COPY --from=frontend /build/web ./web

EXPOSE 8000
CMD ["uvicorn", "backend_app:app", "--host", "0.0.0.0", "--port", "8000"]
