"""
Rekrea image generation service
===============================

A small HTTP wrapper around Hugging Face ``diffusers`` (text-to-image).

Endpoints
---------
GET  /health    Service status, configured model, whether it is loaded in VRAM.
POST /generate  Generate one image; returns ``image/png`` with the used seed in
                the ``X-Seed`` header.
POST /unload    Free the model from VRAM immediately.

Configuration (environment variables)
-------------------------------------
IMAGEGEN_MODEL       Hugging Face model id (default: SD 1.5).
IMAGEGEN_KEEP_ALIVE  Seconds of inactivity before the model is unloaded from
                     VRAM (default 120; 0 keeps it loaded). This lets other
                     GPU users (e.g. Ollama) take the memory back.
IMAGEGEN_MAX_SIZE    Largest allowed width/height in px (default 768).

The pipeline is loaded lazily on the first request, so the container starts
fast and uses no VRAM until it is needed.
"""

import gc
import io
import os
import random
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get("IMAGEGEN_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
KEEP_ALIVE = float(os.environ.get("IMAGEGEN_KEEP_ALIVE", "120"))
MAX_SIZE = int(os.environ.get("IMAGEGEN_MAX_SIZE", "768"))

_lock = threading.Lock()  # one generation at a time; also guards load/unload
_pipe = None
_last_used = 0.0


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    width: int = Field(512, ge=64)
    height: int = Field(512, ge=64)
    steps: int = Field(25, ge=1, le=100)
    guidance_scale: float = Field(7.5, ge=0, le=30)
    seed: Optional[int] = Field(None, ge=0, le=2**32 - 1)


def _load_pipeline():
    """Load the diffusion pipeline onto the GPU (fp16, memory-friendly)."""
    import torch
    from diffusers import StableDiffusionPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the container; check GPU access.")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.enable_attention_slicing()  # lowers peak VRAM at a small speed cost
    return pipe


def _run_pipeline(pipe, req: GenerateRequest, seed: int):
    """Run one generation and return a PIL image."""
    import torch

    generator = torch.Generator("cuda").manual_seed(seed)
    result = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt or None,
        width=req.width,
        height=req.height,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance_scale,
        generator=generator,
    )
    return result.images[0]


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _unload_locked() -> bool:
    global _pipe
    was_loaded = _pipe is not None
    _pipe = None
    _free_gpu()
    return was_loaded


def _idle_watcher() -> None:
    while True:
        time.sleep(5)
        if KEEP_ALIVE <= 0:
            continue
        with _lock:
            if _pipe is not None and time.monotonic() - _last_used > KEEP_ALIVE:
                _unload_locked()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    threading.Thread(target=_idle_watcher, daemon=True).start()
    yield


app = FastAPI(title="Rekrea imagegen", lifespan=_lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "loaded": _pipe is not None, "max_size": MAX_SIZE}


@app.post("/generate")
def generate(req: GenerateRequest):
    global _pipe, _last_used
    for name, value in (("width", req.width), ("height", req.height)):
        if value % 8 or value > MAX_SIZE:
            raise HTTPException(422, f"{name} must be a multiple of 8 and at most {MAX_SIZE}.")

    seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
    with _lock:
        try:
            if _pipe is None:
                _pipe = _load_pipeline()
            image = _run_pipeline(_pipe, req, seed)
        except Exception as exc:  # includes CUDA out-of-memory
            _unload_locked()
            raise HTTPException(500, f"Generation failed: {exc}") from exc
        finally:
            _last_used = time.monotonic()

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"X-Seed": str(seed), "X-Model": MODEL_ID},
    )


@app.post("/unload")
def unload():
    with _lock:
        return {"unloaded": _unload_locked()}
