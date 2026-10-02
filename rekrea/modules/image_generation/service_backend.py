"""
Image generation service backend
================================

Client for the containerised service in ``services/imagegen`` (see
services/README.md). Uses only the standard library.

Configuration (arguments override environment variables):

- ``REKREA_IMAGEGEN_URL``  service URL, default ``http://localhost:8000``
"""

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from .base import GeneratedImage, ImageBackend

DEFAULT_URL = "http://localhost:8000"
# The first request loads the model into VRAM (and downloads it the very first time).
DEFAULT_TIMEOUT = 600.0


class ImageServiceError(RuntimeError):
    """Raised when the image service cannot be reached or returns an error."""


class ServiceBackend(ImageBackend):
    """Text-to-image through the Rekrea imagegen service."""

    def __init__(self, url: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT) -> None:
        url = url or os.environ.get("REKREA_IMAGEGEN_URL", DEFAULT_URL)
        if "://" not in url:
            url = f"http://{url}"
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _open(self, req: urllib.request.Request, timeout: Optional[float] = None):
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ImageServiceError(f"Image service returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise ImageServiceError(
                f"Cannot reach the image service at {self.url} ({exc}). "
                "Is it running? See services/README.md."
            ) from exc

    def health(self) -> dict:
        """Service status: configured model and whether it is loaded in VRAM."""
        with self._open(urllib.request.Request(f"{self.url}/health"), timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def is_available(self) -> bool:
        try:
            return self.health().get("status") == "ok"
        except ImageServiceError:
            return False

    def unload(self) -> None:
        """Ask the service to free its model from VRAM now."""
        req = urllib.request.Request(f"{self.url}/unload", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        with self._open(req, timeout=30):
            pass

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> GeneratedImage:
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
        }
        if seed is not None:
            payload["seed"] = seed
        req = urllib.request.Request(
            f"{self.url}/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._open(req) as resp:
            return GeneratedImage(
                data=resp.read(),
                seed=int(resp.headers["X-Seed"]),
                model=resp.headers.get("X-Model", ""),
            )
