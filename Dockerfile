# booth-gpu — GPU server image.
# Base ships CUDA 12.1 + cuDNN + torch/torchvision (torchvision is needed by
# EasyOCR), so we only layer the detection/render/OCR deps on top.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # EasyOCR model cache lives here; mount a volume to persist it across runs.
    EASYOCR_MODULE_PATH=/models/easyocr

# Minimal OS libs. opencv-python-headless + pymupdf wheels are self-contained,
# so we only need libgomp (OpenMP, used by OpenCV/numpy) and ca-certificates.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CLI/detection deps. cupy + cucim give the fast RAPIDS union-find CCL path
# (--ccl cucim); they match the CUDA 12 base. Pinned-loosely to stay current.
RUN pip install --upgrade pip && pip install \
        "opencv-python-headless>=4.8" \
        "numpy>=1.24" \
        "pymupdf>=1.23" \
        "easyocr>=1.7" \
        "cupy-cuda12x" \
        "cucim-cu12"

# API-layer deps (FastAPI service + Roboflow endpoints). Separate layer so
# editing app code doesn't rebuild the heavy CLI/CUDA layer above.
COPY requirements-api.txt ./requirements-api.txt
RUN pip install -r requirements-api.txt

# App code last so source edits don't bust the dependency layer.
COPY . /app

# Data dirs (overlaid by the bind mount at runtime) + model caches.
RUN mkdir -p /data/in /data/out /data/logs /models/easyocr
ENV PYTHONPATH=/app/server:/app/src:/app \
    LOG_DIR=/data/logs \
    HASH_DB_PATH=/data/hash_db.json \
    PERSIST_IN_DIR=/data/in \
    PERSIST_OUT_DIR=/data/out \
    S3_WRITEBACK_ENABLED=false
VOLUME ["/data", "/models"]

# Default: serve the FastAPI app (same endpoints as prod). For one-off CLI
# detection, override the entrypoint (see docker-compose `booth-cli` service).
EXPOSE 8000
WORKDIR /app/server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
