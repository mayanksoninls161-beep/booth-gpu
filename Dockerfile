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

# Python deps. cupy + cucim give the fast RAPIDS union-find CCL path
# (--ccl cucim); they match the CUDA 12 base. Pinned-loosely to stay current.
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install \
        "opencv-python-headless>=4.8" \
        "numpy>=1.24" \
        "pymupdf>=1.23" \
        "easyocr>=1.7" \
        "cupy-cuda12x" \
        "cucim-cu12"

# App code last so source edits don't bust the dependency layer.
COPY . /app

# Where the user mounts inputs and where outputs are written.
RUN mkdir -p /data /models/easyocr
VOLUME ["/data", "/models"]

# Default outdir -> /data/out so results land on the mounted volume.
ENTRYPOINT ["python", "src/run_real.py"]
CMD ["--help"]
