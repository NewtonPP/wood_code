# Private cloud inference service: model backend + woodchip_core post-processing.
# Build (from repo root):
#   docker build -f inference_service/Dockerfile -t woodchip-inference .
# Run (mock backend — works with no model files):
#   docker run -p 9000:9000 -e INFER_BACKEND=mock woodchip-inference
#
# For real models on a GPU host: switch the base image to an NVIDIA CUDA /
# onnxruntime-gpu base, uncomment onnxruntime-gpu in requirements.txt, then run
# with INFER_BACKEND=onnx and DETR_ONNX_PATH / MOISTURE_ONNX_PATH set.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INFER_BACKEND=mock

# OpenCV (headless) needs a couple of shared libs at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY inference_service/requirements.txt ./inference_service/requirements.txt
RUN pip install --no-cache-dir -r inference_service/requirements.txt

COPY woodchip_core ./woodchip_core
COPY inference_service ./inference_service

EXPOSE 9000
CMD ["uvicorn", "inference_service.main:app", "--host", "0.0.0.0", "--port", "9000"]
