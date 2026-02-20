# Image Model Downloader — One-time container to pre-download HuggingFace models
#
# Usage:
#   docker compose run --rm image_model_downloader
#
# Or standalone:
#   docker run --rm \
#     -v models:/models \
#     -e MODEL_IDS="SG161222/RealVisXL_V4.0_Lightning,black-forest-labs/FLUX.1-dev" \
#     byoc-image-model-downloader

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download.py .

VOLUME /models
ENV MODEL_DIR=/models

CMD ["python3", "download.py"]
