# Rekrea services

Optional, containerised services used by the `rekrea` modules. They are not imported by the core package: modules talk to them over HTTP, so users who never start them need no extra dependencies.

| Service | Port | Used by |
|---|---|---|
| `ollama` | 11434 | `rekrea.modules.prompting` |

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
