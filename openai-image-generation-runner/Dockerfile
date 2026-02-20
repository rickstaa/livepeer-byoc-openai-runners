# BYOC Image Generation Runner
# Supports: FLUX.1-dev, RealVisXL_V4.0_Lightning, and other diffusers models
# Requires: NVIDIA GPU with CUDA (4090/5090 optimized)
#
# Build:
#   docker build -t byoc-image-runner .
#
# Run:
#   docker run --gpus all \
#     -v /path/to/models:/models \
#     -e MODEL_ID=black-forest-labs/FLUX.1-dev \
#     -p 8080:8080 \
#     byoc-image-runner

# --- Base: NVIDIA CUDA 12.8 + Python 3.11 (Ubuntu 22.04) ---
# CUDA 12.8 required for RTX 5090 (Blackwell sm_120), also supports RTX 4090 (Ada sm_89)
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# --- Install PyTorch with CUDA 12.8 support (RTX 4090 + 5090) ---
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# --- Install Python dependencies ---
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# --- Copy application ---
COPY app.py .

# --- Runtime config ---
# Model weights should be mounted as a volume at /models
VOLUME /models

ENV MODEL_DIR=/models
ENV RUNNER_PORT=8080
ENV DEVICE=cuda
ENV DTYPE=float16
ENV MAX_QUEUE_SIZE=5
ENV USE_TORCH_COMPILE=false

# Persistent kernel cache — mount a volume at /cache to survive restarts.
# First startup compiles Triton kernels (slow), subsequent starts reuse cache (fast).
ENV TRITON_CACHE_DIR=/cache/triton
ENV TORCHINDUCTOR_FX_GRAPH_CACHE=1
ENV TORCHINDUCTOR_CACHE_DIR=/cache/inductor

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')" || exit 1

CMD ["python3", "app.py"]
