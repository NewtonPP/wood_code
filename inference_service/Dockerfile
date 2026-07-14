# Private cloud inference service: ONNX models + woodchip_core post-processing.
# Build (from repo root; requires inference_service/models/detr_resnet101.onnx
# to exist — export it first, see DEPLOYMENT.md):
#   docker build -f inference_service/Dockerfile -t woodchip-inference .
# Run:
#   docker run -p 9000:9000 woodchip-inference
#
# For GPU inference: switch the base image to an NVIDIA CUDA / onnxruntime-gpu
# base and swap onnxruntime for onnxruntime-gpu in requirements.txt.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

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
