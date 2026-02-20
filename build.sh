#!/usr/bin/env bash
set -euo pipefail

# Build all Docker images for the BYOC OpenAI-compatible runners
#
# Environment:
#   REGISTRY            - Docker registry prefix (optional, e.g. "myregistry.io")
#   IMAGE_REGISTRY      - Separate registry for the image generation runner (optional,
#                          falls back to REGISTRY). Useful when your primary registry
#                          has upload size limits.
#   TAG                 - Image tag (default: latest)
#   PUSH                - Push images after build (default: false)

TAG="${TAG:-latest}"
PUSH="${PUSH:-false}"

# If REGISTRY is set, prefix image names with it (adds trailing slash)
PREFIX="${REGISTRY:+${REGISTRY}/}"

# IMAGE_REGISTRY allows a separate registry for the large image generation runner
IMAGE_PREFIX="${IMAGE_REGISTRY:+${IMAGE_REGISTRY}/}"
IMAGE_PREFIX="${IMAGE_PREFIX:-${PREFIX}}"

images=(
  # context_dir                       image_name
  "./openai-chat-completion-runner    ${PREFIX}livepeer-byoc-openai-chat-completion-runner:${TAG}"
  "./openai-image-generation-runner   ${IMAGE_PREFIX}livepeer-byoc-openai-image-generation-runner:${TAG}"
  "./openai-embeddings-runner         ${PREFIX}livepeer-byoc-openai-embeddings-runner:${TAG}"
  "./image-model-downloader            ${PREFIX}livepeer-byoc-openai-image-model-downloader:${TAG}"
)

for entry in "${images[@]}"; do
  context=$(echo "$entry" | awk '{print $1}')
  image=$(echo "$entry" | awk '{print $2}')
  echo "==> Building ${image} from ${context}"
  docker build -t "$image" "$context"
  echo ""
done

echo "All images built successfully."

if [ "$PUSH" = "true" ]; then
  echo ""
  echo "Pushing images..."
  for entry in "${images[@]}"; do
    image=$(echo "$entry" | awk '{print $2}')
    echo "==> Pushing ${image}"
    docker push "$image"
  done
  echo ""
  echo "All images pushed successfully."
fi
