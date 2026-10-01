"""
Ollama prompting backend
========================

Talks to an Ollama server over its HTTP API (https://github.com/ollama/ollama).
Uses only the standard library, so the module adds no dependencies.

Start the server with ``docker compose -f services/docker-compose.yml up -d``
and pull a model, e.g. ``docker exec rekrea-ollama ollama pull llama3.2:3b``.

Configuration (arguments override environment variables):

- ``OLLAMA_HOST``         server URL, default ``http://localhost:11434``
- ``REKREA_OLLAMA_MODEL`` model name, default :data:`DEFAULT_MODEL`
"""

import json
import os
import urllib.error
import urllib.request
from typing import List, Optional

from .base import PromptBackend

DEFAULT_HOST = "http://localhost:11434"
# ~3B parameter model: about 2 GB, fits alongside a desktop on an 8 GB GPU.
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_TIMEOUT = 120.0


class OllamaError(RuntimeError):
    """Raised when the Ollama server cannot be reached or returns an error."""


class OllamaBackend(PromptBackend):
    """Text generation through a local Ollama server."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model or os.environ.get("REKREA_OLLAMA_MODEL", DEFAULT_MODEL)
        host = host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
        if "://" not in host:
            host = f"http://{host}"
        self.host = host.rstrip("/")
        self.timeout = timeout

    # -- HTTP helper --------------------------------------------------------

    def _request(self, path: str, payload: Optional[dict] = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.host} ({exc}). Is the server running? "
                "See services/README.md."
            ) from exc

    # -- Public API ---------------------------------------------------------

    def list_models(self) -> List[str]:
        """Names of the models available on the server."""
        return [m["name"] for m in self._request("/api/tags").get("models", [])]

    def is_available(self) -> bool:
        try:
            names = self.list_models()
        except OllamaError:
            return False
        # Ollama reports "name:tag"; accept a bare name as the ":latest" tag.
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return wanted in names

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        payload: dict = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if seed is not None:
            options["seed"] = seed
        if options:
            payload["options"] = options

        result = self._request("/api/generate", payload)
        if "response" not in result:
            raise OllamaError(f"Unexpected Ollama response: {result}")
        return result["response"].strip()
