#!/usr/bin/env bash
set -euo pipefail

# Setup script for new machines — downloads model weights and pre-compiles
# Triton/Inductor kernels so the runners start fast.
#
# Safe to re-run: already-downloaded models are skipped, existing kernel
# caches are reused.
#
# Usage:
#   ./setup-models.sh                          # download + warmup default model (RealVisXL)
#   MODEL_IDS="black-forest-labs/FLUX.1-dev" ./setup-models.sh   # use FLUX.1-dev instead
#   HF_TOKEN=hf_xxx ./setup-models.sh          # for gated models (e.g. FLUX.1-dev)
#
# Environment:
#   REGISTRY        - Docker registry prefix (optional, e.g. "myregistry.io")
#   IMAGE_REGISTRY  - Separate registry for the image generation runner (optional, falls back to REGISTRY)
#   TAG             - Image tag (default: latest)

TAG="${TAG:-latest}"

# If REGISTRY is set, prefix image names with it (adds trailing slash)
PREFIX="${REGISTRY:+${REGISTRY}/}"

# IMAGE_REGISTRY allows a separate registry for the large image generation runner
IMAGE_PREFIX="${IMAGE_REGISTRY:+${IMAGE_REGISTRY}/}"
IMAGE_PREFIX="${IMAGE_PREFIX:-${PREFIX}}"

DOWNLOADER_IMAGE="${PREFIX}livepeer-byoc-openai-image-model-downloader:${TAG}"
RUNNER_IMAGE="${IMAGE_PREFIX}livepeer-byoc-openai-image-generation-runner:${TAG}"

MODEL_IDS="${MODEL_IDS:-SG161222/RealVisXL_V4.0_Lightning}"
MODEL_DIR="/models"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
HF_TOKEN="${HF_TOKEN:-}"

# ---------------------------------------------------------------------------
# 1. Create volumes (if they don't exist)
# ---------------------------------------------------------------------------
echo "==> Creating Docker volumes (if needed)"
docker volume create ai-image-models 2>/dev/null || true
docker volume create ai-image-kernel_cache 2>/dev/null || true
echo ""

# ---------------------------------------------------------------------------
# 2. Download image model weights (skips already-downloaded models)
# ---------------------------------------------------------------------------
echo "==> Downloading image model weights: ${MODEL_IDS}"
echo "    (already-downloaded models will be verified and skipped)"

HF_TOKEN_ARGS=()
if [ -n "$HF_TOKEN" ]; then
  HF_TOKEN_ARGS=(-e "HF_TOKEN=${HF_TOKEN}")
fi

docker run --rm \
  -v ai-image-models:${MODEL_DIR} \
  -e "MODEL_IDS=${MODEL_IDS}" \
  -e "MODEL_DIR=${MODEL_DIR}" \
  "${HF_TOKEN_ARGS[@]+"${HF_TOKEN_ARGS[@]}"}" \
  "$DOWNLOADER_IMAGE"

echo ""
echo "==> Image model download complete."
echo ""

# ---------------------------------------------------------------------------
# 3. Pre-compile Triton kernels for each image model on this GPU
# ---------------------------------------------------------------------------
echo "==> Pre-compiling Triton kernels for each image model..."
echo "    (first compilation is slow — subsequent runs reuse the cache)"
echo ""

IFS=',' read -ra MODELS <<< "$MODEL_IDS"
for model_id in "${MODELS[@]}"; do
  model_id=$(echo "$model_id" | xargs)  # trim whitespace
  echo "==> Warming up kernels for: ${model_id}"

  docker run --rm --gpus all \
    -v ai-image-models:${MODEL_DIR} \
    -v ai-image-kernel_cache:/cache \
    -e "WARMUP_ONLY=true" \
    -e "MODEL_ID=${model_id}" \
    -e "MODEL_DIR=${MODEL_DIR}" \
    -e "DEVICE=${DEVICE}" \
    -e "DTYPE=${DTYPE}" \
    -e "USE_TORCH_COMPILE=true" \
    "$RUNNER_IMAGE"

  echo ""
  echo "==> Kernel warmup complete for: ${model_id}"
  echo ""
done

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "============================================"
echo "  Setup complete!"
echo "  Image models downloaded: ${MODEL_IDS}"
echo "  Kernels compiled for: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown GPU')"
echo "============================================"
