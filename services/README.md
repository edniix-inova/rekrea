# Rekrea services

Optional, containerised services used by the `rekrea` modules. They are not imported by the core package: modules talk to them over HTTP, so users who never start them need no extra dependencies.

| Service | Port | Used by |
|---|---|---|
| `ollama` | 11434 | `rekrea.modules.prompting` |
| `imagegen` | 8000 | `rekrea.modules.image_generation` |

## Ollama

```bash
docker compose -f services/docker-compose.yml up -d

# Pull a model (once). Pick one that fits your VRAM; ~3B models need about 2-3 GB.
docker exec rekrea-ollama ollama pull llama3.2:3b

# Check it works
curl http://localhost:11434/api/tags
```

Notes:
- The port is bound to `127.0.0.1` only.
- Models are stored in the `ollama_models` Docker volume.
- The model is unloaded from VRAM after `OLLAMA_KEEP_ALIVE` (default `2m`) of inactivity, so the GPU can be reused by other steps. Override with `OLLAMA_KEEP_ALIVE=10m docker compose ... up -d`.
- GPU check: `docker run --rm --gpus all ubuntu nvidia-smi`.

## Image generation (`imagegen`)

A small FastAPI wrapper around Hugging Face `diffusers`. The default model is Stable Diffusion 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`), which fits in roughly 4 GB of free VRAM at 512 px. It is released under the CreativeML OpenRAIL-M licence; check the model card before using outputs commercially.

```bash
docker compose -f services/docker-compose.yml up -d --build

# Check it (the model loads on the first /generate request, not at startup)
curl http://localhost:8000/health

# Generate (the first call downloads ~4 GB of weights into the hf_cache volume)
python scripts/image_generation_pipeline.py --prompt "a red fox in snow" --seed 7
```

API: `GET /health`, `POST /generate` (returns `image/png`, seed in the `X-Seed` header), `POST /unload`.

Notes:
- **Shared GPU:** the model is loaded on demand and dropped from VRAM after `IMAGEGEN_KEEP_ALIVE` seconds idle (default `120`; `0` keeps it loaded). `POST /unload` frees it immediately. Ollama unloads after `OLLAMA_KEEP_ALIVE`, and `scripts/image_generation_pipeline.py` asks Ollama to unload before generating.
- **Limits:** width and height must be multiples of 8 and at most `IMAGEGEN_MAX_SIZE` (default 768), to protect small GPUs from out-of-memory errors. After a failed generation the model is unloaded so the next request starts clean.
- **Another model:** set `IMAGEGEN_MODEL` to a Stable Diffusion 1.x-class Hugging Face id. SDXL-class models need a different pipeline and more VRAM, and are not supported yet.
- **Reproducibility:** the same prompt, seed, size, steps and model give the same image on the same setup. Each run saves a JSON next to the PNG with these values.

### Troubleshooting: CUDA does not initialise in the container

If this fails with `Error 500: named symbol not found` (or similar):

```bash
docker exec rekrea-imagegen python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

the CUDA runtime bundled with torch may be too new for the host driver or WSL libraries. Rebuild with a torch build on an older CUDA 12.x runtime:

```bash
# pick an index that has a torch build for your Python (3.11); cu126 and cu128 are common
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
  docker compose -f services/docker-compose.yml build --no-cache imagegen
docker compose -f services/docker-compose.yml up -d
```
