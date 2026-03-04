#!/usr/bin/env python3
"""
BYOC Image Generation Runner — OpenAI-compatible /v1/images/generations endpoint.

Loads a single diffusers model at startup (configurable via MODEL_ID env var),
keeps it warm on GPU, and serves OpenAI-format image generation requests.

Supported models:
  - black-forest-labs/FLUX.1-dev (FLUX pipeline)
  - SG161222/RealVisXL_V4.0_Lightning (SDXL pipeline)
  - Any HuggingFace diffusers-compatible text-to-image model

Environment variables:
  MODEL_ID           - HuggingFace model ID (required)
  MODEL_DIR          - Local directory for model weights (default: /models)
  RUNNER_PORT        - HTTP port (default: 8080)
  MAX_QUEUE_SIZE     - Max queued requests before 429 (default: 5)
  DEVICE             - torch device (default: cuda)
  DTYPE              - torch dtype: float16 | bfloat16 | float32 (default: float16)
  USE_TORCH_COMPILE  - Enable torch.compile() for inference speedup (default: true)
  DEFAULT_WIDTH      - Default image width (default: 1024)
  DEFAULT_HEIGHT     - Default image height (default: 1024)
  DEFAULT_STEPS      - Default inference steps (default: model-dependent)
  DEFAULT_GUIDANCE   - Default guidance scale (default: model-dependent)
"""

import asyncio
import base64
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("MODEL_ID", "")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
RUNNER_PORT = int(os.environ.get("RUNNER_PORT", "8080"))
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "5"))
DEVICE = os.environ.get("DEVICE", "cuda")
DTYPE_STR = os.environ.get("DTYPE", "float16")
USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "true").lower() in ("true", "1", "yes")
DEFAULT_WIDTH = int(os.environ.get("DEFAULT_WIDTH", "1024"))
DEFAULT_HEIGHT = int(os.environ.get("DEFAULT_HEIGHT", "1024"))
DEFAULT_STEPS = os.environ.get("DEFAULT_STEPS", "")
DEFAULT_GUIDANCE = os.environ.get("DEFAULT_GUIDANCE", "")

DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("image-runner")

# ---------------------------------------------------------------------------
# Model profiles — sensible defaults per model family
# ---------------------------------------------------------------------------

MODEL_PROFILES = {
    "FLUX": {
        "default_steps": 28,
        "default_guidance": 3.5,
        "scheduler": None,  # FLUX uses its built-in FlowMatchEulerDiscreteScheduler
    },
    "RealVisXL": {
        "default_steps": 6,  # Lightning variant is fast
        "default_guidance": 1.5,
        "scheduler": None,  # Uses DPM++ SDE Karras by default
    },
    "SDXL": {
        "default_steps": 20,
        "default_guidance": 7.5,
        "scheduler": None,
    },
}


def detect_model_family(model_id: str) -> str:
    """Detect model family from model ID for default parameters."""
    lower = model_id.lower()
    if "flux" in lower:
        return "FLUX"
    if "realvis" in lower or "lightning" in lower:
        return "RealVisXL"
    if "sdxl" in lower or "stable-diffusion-xl" in lower:
        return "SDXL"
    return "SDXL"  # safe default


def get_default_steps(model_id: str) -> int:
    if DEFAULT_STEPS:
        return int(DEFAULT_STEPS)
    family = detect_model_family(model_id)
    return MODEL_PROFILES.get(family, MODEL_PROFILES["SDXL"])["default_steps"]


def get_default_guidance(model_id: str) -> float:
    if DEFAULT_GUIDANCE:
        return float(DEFAULT_GUIDANCE)
    family = detect_model_family(model_id)
    return MODEL_PROFILES.get(family, MODEL_PROFILES["SDXL"])["default_guidance"]


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

_pipeline = None
_model_family = None
_semaphore: Optional[asyncio.Semaphore] = None


def load_pipeline(model_id: str, model_dir: str, device: str, dtype: torch.dtype):
    """Load the appropriate diffusers pipeline based on model family."""
    global _pipeline, _model_family

    _model_family = detect_model_family(model_id)
    logger.info(f"Loading model: {model_id} (family={_model_family}, dtype={dtype}, device={device})")
    logger.info(f"Model cache directory: {model_dir}")

    start = time.time()

    if _model_family == "FLUX":
        from diffusers import FluxPipeline

        _pipeline = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=model_dir,
        )
        # FLUX.1-dev is ~24GB in fp16 — .to(device) would OOM on 32GB GPUs
        # because text encoders + VAE push total past VRAM capacity.
        # CPU offload keeps components in RAM, moving each to GPU only during
        # its forward pass, keeping peak VRAM at ~12-16GB.
        _pipeline.enable_model_cpu_offload()

    elif _model_family in ("RealVisXL", "SDXL"):
        from diffusers import StableDiffusionXLPipeline

        _pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=model_dir,
            use_safetensors=True,
        )
        _pipeline.to(device)

    else:
        # Generic AutoPipeline fallback
        from diffusers import AutoPipelineForText2Image

        _pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=model_dir,
        )
        _pipeline.to(device)

    # Performance optimizations
    # 1. Enable memory-efficient attention if available
    try:
        _pipeline.enable_xformers_memory_efficient_attention()
        logger.info("xformers memory-efficient attention enabled")
    except Exception:
        logger.info("xformers not available, using default attention")

    # 2. Enable VAE slicing for lower VRAM on large batches
    if hasattr(_pipeline, "enable_vae_slicing"):
        _pipeline.enable_vae_slicing()
        logger.info("VAE slicing enabled")

    # 3. Enable VAE tiling for very large images
    if hasattr(_pipeline, "enable_vae_tiling"):
        _pipeline.enable_vae_tiling()
        logger.info("VAE tiling enabled")

    # 4. torch.compile() for inference speedup on 4090/5090 (Ampere+)
    triton_cache = os.environ.get("TRITON_CACHE_DIR", "")
    if triton_cache and os.path.isdir(triton_cache) and os.listdir(triton_cache):
        logger.info(f"Triton kernel cache found at {triton_cache} — reusing compiled kernels")
    elif triton_cache:
        logger.info(f"Triton kernel cache empty at {triton_cache} — first run will compile kernels")
    #    "default" mode is the safest — basic Triton kernel fusion without
    #    autotuning or CUDA graphs. Stable across GPU architectures including
    #    Blackwell (5090). Override with TORCH_COMPILE_MODE env var:
    #      "max-autotune-no-cudagraphs" — benchmarks kernel variants (faster steady-state, slower compile)
    #      "reduce-overhead" — adds CUDA graphs (fastest steady-state, slowest first-run)
    compile_mode = os.environ.get("TORCH_COMPILE_MODE", "default")
    if USE_TORCH_COMPILE and device == "cuda":
        try:
            if hasattr(_pipeline, "unet"):
                logger.info(f"Compiling unet with torch.compile(mode='{compile_mode}')...")
                _pipeline.unet = torch.compile(
                    _pipeline.unet, mode=compile_mode
                )

            # FLUX uses a transformer, not unet
            if hasattr(_pipeline, "transformer"):
                logger.info(f"Compiling transformer with torch.compile(mode='{compile_mode}')...")
                _pipeline.transformer = torch.compile(
                    _pipeline.transformer, mode=compile_mode
                )
            logger.info("torch.compile() applied for inference acceleration")
        except Exception as e:
            logger.warning(f"torch.compile() failed (non-fatal): {e}")

    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s")

    # Warmup pass — first inference is always slower due to CUDA kernels.
    # This also triggers actual torch.compile() compilation (which is lazy),
    # so we catch failures and fall back to eager mode if needed.
    if USE_TORCH_COMPILE:
        logger.info("Running warmup inference (first run triggers torch.compile — this may take several minutes)...")
    else:
        logger.info("Running warmup inference...")
    warmup_start = time.time()
    try:
        with torch.inference_mode():
            _ = _pipeline(
                prompt="warmup",
                width=512,
                height=512,
                num_inference_steps=2,
                guidance_scale=1.0,
                output_type="latent",
            )
        torch.cuda.synchronize()
        logger.info(f"Warmup complete in {time.time() - warmup_start:.1f}s")
    except Exception as e:
        logger.warning(f"Warmup failed with torch.compile(), falling back to eager mode: {e}")
        # Restore uncompiled components and retry
        if hasattr(_pipeline, "unet") and hasattr(_pipeline.unet, "_orig_mod"):
            _pipeline.unet = _pipeline.unet._orig_mod
        if hasattr(_pipeline, "transformer") and hasattr(_pipeline.transformer, "_orig_mod"):
            _pipeline.transformer = _pipeline.transformer._orig_mod
        torch.cuda.empty_cache()
        warmup_start = time.time()
        with torch.inference_mode():
            _ = _pipeline(
                prompt="warmup",
                width=512,
                height=512,
                num_inference_steps=2,
                guidance_scale=1.0,
                output_type="latent",
            )
        torch.cuda.synchronize()
        logger.info(f"Warmup complete (eager mode) in {time.time() - warmup_start:.1f}s")


# ---------------------------------------------------------------------------
# Request / Response models (OpenAI format)
# ---------------------------------------------------------------------------

class ImageGenerationRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    n: int = Field(default=1, ge=1, le=10)
    size: Optional[str] = None  # "1024x1024"
    response_format: Optional[str] = "b64_json"  # "url" or "b64_json"
    quality: Optional[str] = None  # "standard" or "hd"
    # Extended parameters (non-OpenAI, but useful)
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None


class ImageData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global _semaphore

    if not MODEL_ID:
        logger.error("MODEL_ID environment variable is required")
        raise RuntimeError("MODEL_ID is required")

    dtype = DTYPE_MAP.get(DTYPE_STR.lower(), torch.float16)
    load_pipeline(MODEL_ID, MODEL_DIR, DEVICE, dtype)

    _semaphore = asyncio.Semaphore(MAX_QUEUE_SIZE)

    logger.info(f"Image runner ready — model={MODEL_ID}, queue_size={MAX_QUEUE_SIZE}")
    yield

    # Cleanup
    logger.info("Shutting down, releasing GPU memory...")
    global _pipeline
    del _pipeline
    torch.cuda.empty_cache()


app = FastAPI(title="BYOC Image Runner", lifespan=lifespan)


def parse_size(size: Optional[str]) -> tuple[int, int]:
    """Parse '1024x1024' into (width, height)."""
    if not size:
        return DEFAULT_WIDTH, DEFAULT_HEIGHT
    try:
        parts = size.lower().split("x")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return DEFAULT_WIDTH, DEFAULT_HEIGHT


def generate_images_sync(req: ImageGenerationRequest) -> list[bytes]:
    """Run inference on GPU (blocking). Returns list of PNG byte buffers."""
    width, height = parse_size(req.size)
    steps = req.num_inference_steps or get_default_steps(MODEL_ID)
    guidance = req.guidance_scale if req.guidance_scale is not None else get_default_guidance(MODEL_ID)

    # HD quality = more steps
    if req.quality == "hd":
        steps = max(steps, steps * 2)

    generator = None
    if req.seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(req.seed)

    kwargs = {
        "prompt": req.prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "num_images_per_prompt": req.n,
        "generator": generator,
    }

    # Add negative prompt for SDXL-based models
    if req.negative_prompt and _model_family in ("RealVisXL", "SDXL"):
        kwargs["negative_prompt"] = req.negative_prompt

    logger.info(
        f"Generating {req.n} image(s): {width}x{height}, steps={steps}, "
        f"guidance={guidance}, seed={req.seed}"
    )

    start = time.time()
    with torch.inference_mode():
        result = _pipeline(**kwargs)

    elapsed = time.time() - start
    logger.info(f"Generation complete in {elapsed:.1f}s ({req.n} images)")

    # Convert PIL images to PNG bytes
    png_buffers = []
    for img in result.images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_buffers.append(buf.getvalue())

    return png_buffers


@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def create_image(req: ImageGenerationRequest):
    """OpenAI-compatible image generation endpoint."""

    # Check queue capacity
    if _semaphore.locked() and _semaphore._value == 0:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": f"Server busy — max queue size ({MAX_QUEUE_SIZE}) reached. Try again later.",
                    "type": "rate_limit_error",
                }
            },
        )

    async with _semaphore:
        loop = asyncio.get_event_loop()
        try:
            png_buffers = await loop.run_in_executor(None, generate_images_sync, req)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(
                status_code=507,
                detail={
                    "error": {
                        "message": "GPU out of memory. Try a smaller size or fewer images.",
                        "type": "server_error",
                    }
                },
            )
        except Exception as e:
            logger.exception("Image generation failed")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": f"Generation failed: {str(e)}",
                        "type": "server_error",
                    }
                },
            )

    # Build response
    data = []
    for png_bytes in png_buffers:
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        data.append(ImageData(b64_json=b64))

    return ImageGenerationResponse(
        created=int(time.time()),
        data=data,
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    warmup_only = os.environ.get("WARMUP_ONLY", "").lower() in ("true", "1", "yes")

    if warmup_only:
        # Pre-warm mode: load model, compile kernels, populate cache, then exit.
        logger.info("WARMUP_ONLY mode — compiling kernels and populating cache...")
        dtype = DTYPE_MAP.get(DTYPE_STR.lower(), torch.float16)
        load_pipeline(MODEL_ID, MODEL_DIR, DEVICE, dtype)
        logger.info("Kernel cache populated. Exiting.")
    else:
        import uvicorn

        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=RUNNER_PORT,
            log_level="info",
            workers=1,  # Single worker — GPU is the bottleneck
        )
