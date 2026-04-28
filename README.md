# Livepeer BYOC OpenAI-Compatible Runners

This repo contains the **BYOC runners** that handle OpenAI-compatible inference requests routed through the Livepeer network. LLM chat completions and embeddings forward to an upstream **Ollama** instance, while image generation runs locally on GPU via diffusers.

**Supported endpoints:**
- `POST /v1/chat/completions` (streaming supported)
- `POST /v1/images/generations`
- `POST /v1/embeddings`

This project assumes you already have a running **Livepeer Orchestrator**, **Gateway**, and [**Gateway Proxy**](https://github.com/Cloud-SPE/livepeer-byoc-gateway-proxy). The runners register their capabilities with the orchestrator using the [register-capability](https://github.com/Cloud-SPE/livepeer-byoc-register-capabilities) service.

---

## Architecture

```
Client (OpenAI SDK)
│
├─ /v1/chat/completions ──→ Gateway Proxy ──→ Livepeer Gateway ──→ Chat Completion Runner ──→ Ollama
├─ /v1/images/generations ─→ Gateway Proxy ──→ Livepeer Gateway ──→ Image Generation Runner ──→ diffusers (GPU)
└─ /v1/embeddings ─────────→ Gateway Proxy ──→ Livepeer Gateway ──→ Embeddings Runner ──→ Ollama
```

### Component roles

| Component | Language | What it does |
|-----------|----------|-------------|
| **Chat Completion Runner** | Go | Forwards `/v1/chat/completions` to upstream Ollama with SSE passthrough |
| **Image Generation Runner** | Python | Loads a diffusers model on GPU at startup, serves `/v1/images/generations` locally |
| **Embeddings Runner** | Go | Forwards `/v1/embeddings` to upstream Ollama. Synchronous JSON request/response |

External components (not in this repo):
- [**Gateway Proxy**](https://github.com/Cloud-SPE/livepeer-byoc-gateway-proxy) — Routes requests by path, injects Livepeer BYOC headers
- [**Register Capability**](https://github.com/Cloud-SPE/livepeer-byoc-register-capabilities) — Registers runner capabilities with the orchestrator

LLM requests are **byte-for-byte streaming passthrough** (SSE). Image requests run GPU inference locally with a configurable request queue. Embeddings are synchronous JSON request/response.

---

## What's included

- Chat completion runner (`/v1/chat/completions` → Ollama)
- Image generation runner (`/v1/images/generations` → diffusers on GPU)
- Embeddings runner (`/v1/embeddings` → Ollama)
- Image model downloader (pre-downloads HuggingFace diffusers weights)
- Setup script (`setup-models.sh`) for model download + GPU kernel warmup
- ESM-only JS tester using the official OpenAI SDK

---

## Repository layout

```
openai-runners/
├── build.sh                        # Build all Docker images
├── setup-models.sh                 # New machine setup (download weights + compile kernels)
├── docker-compose.yml              # Runners + capability registration
├── docker-compose.ollama.yml       # Reference Ollama setup
├── openai-chat-completion-runner/  # BYOC chat completion runner (Go) — Ollama upstream
│   ├── main.go
│   ├── go.mod
│   └── Dockerfile
├── openai-image-generation-runner/ # BYOC image generation runner (Python) — GPU inference
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── openai-embeddings-runner/       # BYOC embeddings runner (Go) — Ollama upstream
│   ├── main.go
│   ├── go.mod
│   └── Dockerfile
├── image-model-downloader/         # Pre-downloads HuggingFace diffusers models
│   ├── download.py
│   ├── requirements.txt
│   └── Dockerfile
└── openai-tester/                  # ESM-only OpenAI SDK tester
    ├── package.json
    ├── test-chat-completion.mjs
    ├── test-image-generation.mjs
    └── test-text-embedding.mjs
```

---

## Prerequisites

- Docker + Docker Compose
- NVIDIA GPU with CUDA support (4090/5090 recommended for image generation)
- `nvidia-container-toolkit` installed on the host
- A running Livepeer **Orchestrator** and **Gateway**
- A running [**Gateway Proxy**](https://github.com/Cloud-SPE/livepeer-byoc-gateway-proxy)
- An Ollama instance with an OpenAI-compatible endpoint:
  - `POST /v1/chat/completions` (supports SSE streaming)
  - `POST /v1/embeddings`

---

## Environment variables

### Chat Completion Runner (`openai_chat_completion_runner`)

| Variable | Default | Description |
|----------|---------|-------------|
| `RUNNER_ADDR` | `:8080` | Listen address |
| `UPSTREAM_URL` | **required** | Ollama OpenAI-compatible endpoint, e.g. `http://HOST:PORT/v1/chat/completions` |
| `MAX_BODY_BYTES` | `26214400` (25 MiB) | Max request body size. Oversized requests get a `413` (no silent truncation). Raise for very large multimodal payloads. |

### Embeddings Runner (`byoc_embeddings_runner`)

| Variable | Default | Description |
|----------|---------|-------------|
| `RUNNER_ADDR` | `:8080` | Listen address |
| `UPSTREAM_URL` | **required** | Ollama OpenAI-compatible endpoint, e.g. `http://HOST:PORT/v1/embeddings` |

### Image Generation Runner (`byoc_image_runner`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | **required** | HuggingFace model ID (e.g. `SG161222/RealVisXL_V4.0_Lightning` or `black-forest-labs/FLUX.1-dev`) |
| `MODEL_DIR` | `/models` | Directory for model weights (mount as Docker volume) |
| `RUNNER_PORT` | `8080` | Listen port |
| `DEVICE` | `cuda` | Compute device |
| `DTYPE` | `float16` | Data type (`float16`, `bfloat16`, `float32`) |
| `MAX_QUEUE_SIZE` | `5` | Max queued requests before 429 |
| `USE_TORCH_COMPILE` | `true` | Enable torch.compile() for acceleration |

### Register Capability

Uses [`registry.livepeer.tools/livepeer-byoc-register-capability:latest`](https://github.com/Cloud-SPE/livepeer-byoc-register-capabilities).

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCH_URL` | — | Orchestrator URL (e.g. `https://YOUR_ORCHESTRATOR:8935`) |
| `ORCH_SECRET` | — | Orchestrator secret |
| `CAPABILITY_NAME` | — | Capability to register (e.g. `openai-chat-completion`) |
| `CAPABILITY_URL` | — | Runner URL (e.g. `http://openai_chat_completion_runner:8080`) |
| `CAPACITY` | `5` | Max concurrent requests |
| `PRICE_PER_UNIT` | `250` | Price per unit |

---

## Quick start

### 1) Build images

```bash
./build.sh

# Or build and push to a custom registry
REGISTRY="myregistry.io" PUSH=true ./build.sh
```

### 2) Setup models (new machine)

```bash
# Download default model (RealVisXL) + pre-compile GPU kernels
./setup-models.sh

# Use FLUX.1-dev instead (requires HF token for gated model)
MODEL_IDS="black-forest-labs/FLUX.1-dev" HF_TOKEN=hf_xxx ./setup-models.sh

# Multiple models
MODEL_IDS="SG161222/RealVisXL_V4.0_Lightning,black-forest-labs/FLUX.1-dev" HF_TOKEN=hf_xxx ./setup-models.sh
```

### 3) Start Ollama (if needed)

A reference `docker-compose.ollama.yml` is included. If you don't have an existing Ollama instance:

```bash
docker compose -f docker-compose.ollama.yml up -d

# Pull the models you need
docker exec ollama ollama pull qwen3:8b
docker exec ollama ollama pull nomic-embed-text
```

The runners default to `http://ollama:11434`. If your Ollama is elsewhere, update `UPSTREAM_URL` in `docker-compose.yml`.

### 4) Configure

Edit `docker-compose.yml` — set `ORCH_URL` and `ORCH_SECRET` in the register services to point at your orchestrator:

```yaml
register_capability:
  environment:
    - ORCH_URL=https://YOUR_ORCHESTRATOR:8935
    - ORCH_SECRET=your-secret
```

### 5) Start runners

```bash
docker compose up --build
```

The runners start and the register containers automatically register each capability with the orchestrator.

---

## API endpoints

All endpoints are available through your [Gateway Proxy](https://github.com/Cloud-SPE/livepeer-byoc-gateway-proxy) (default `http://localhost:8090`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | LLM chat (streaming supported) |
| `/v1/images/generations` | POST | Image generation |
| `/v1/embeddings` | POST | Text embeddings |
| `/healthz` | GET | Proxy health check |

---

## Testing

### curl examples

**Chat completion (non-streaming):**
```bash
curl -sS http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:8b",
    "stream": false,
    "messages": [{"role":"user","content":"Say hello in one sentence."}]
  }'
```

**Chat completion (streaming):**
```bash
curl -N http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:8b",
    "stream": true,
    "messages": [{"role":"user","content":"Count from 1 to 10 slowly."}]
  }'
```

**Image generation:**
```bash
curl -sS http://localhost:8090/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SG161222/RealVisXL_V4.0_Lightning",
    "prompt": "A cool cat on the beach, digital art",
    "n": 1,
    "size": "1024x1024"
  }'
```

**Embeddings:**
```bash
curl -sS http://localhost:8090/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text",
    "input": "Hello world"
  }'
```

### Vision (`image_url`)

`image_url` content forwards byte-for-byte to the upstream. Ollama's OpenAI shim (default upstream) accepts only base64 `data:` URLs; vLLM and OpenAI's API accept HTTP URLs too. Use the object form `image_url: {"url": "..."}` for portability.

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8090/v1", api_key="not-used")
b64 = base64.b64encode(open("cat.jpg", "rb").read()).decode()

print(client.chat.completions.create(
    model="gemma3:4b",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]}],
).choices[0].message.content)
```

Base64 inflates ~33%; resize images before encoding (vision models tile at ~512–768 px). `MAX_BODY_BYTES` default 25 MiB fits a single ~18 MB image; oversized payloads return `413`.

### OpenAI JavaScript SDK

```bash
cd openai-tester
npm install

# Chat completion (non-streaming + streaming)
OPENAI_BASE_URL="http://localhost:8090/v1" MODEL="qwen3:8b" npm run test:chat-completion

# Image generation
OPENAI_BASE_URL="http://localhost:8090/v1" MODEL="SG161222/RealVisXL_V4.0_Lightning" npm run test:image-generation

# Text embedding
OPENAI_BASE_URL="http://localhost:8090/v1" MODEL="nomic-embed-text" npm run test:text-embedding
```

> Note: The SDK requires an `apiKey` string. This project does not use it — the proxy strips any incoming `Authorization` header.

---

## Troubleshooting

### Capability not registered

- Check logs from `register_capability` container (prints "capability registered" on success)
- Check orchestrator logs for `/capability/register` requests
- Ensure `ORCH_SECRET` matches `-orchSecret` on orchestrator

### 404 on gateway BYOC path

Confirm the gateway is listening and the proxy is configured to route to it.

### Streaming not streaming (buffered output)

- Ensure any reverse proxy disables response buffering for SSE
- Both Go services disable forced HTTP/2 and flush frequently
- Use `curl -N` for streaming tests

### Upstream errors from Ollama

Verify connectivity from inside the runner container:

```bash
docker exec -it openai_chat_completion_runner sh
# then:
wget -qO- http://YOUR_OLLAMA_SERVER:11434/v1/chat/completions || true
```

---

## Security notes

- This project assumes **Traefik (or equivalent)** is the enforcement point for auth, rate limiting, and IP allowlists.
- The proxy intentionally strips `Authorization` before calling the gateway.
- The register container uses `InsecureSkipVerify: true` to mirror the BYOC docs for local compose. For production, use valid TLS certs and remove insecure TLS.

---

## License

This project is licensed under the [MIT License](LICENSE).
