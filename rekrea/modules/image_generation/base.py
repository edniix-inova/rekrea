"""
Image generation backend interface
==================================

An image backend turns a text prompt into an image. Implementations may call a
service (see :mod:`.service_backend`) or run a model in-process.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class GeneratedImage:
    """A generated image (PNG bytes) with the settings that reproduce it."""

    data: bytes
    seed: int
    model: str

    def save(self, path: Union[str, Path]) -> Path:
        """Write the PNG to *path*, creating parent folders. Returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.data)
        return path


class ImageBackend(ABC):
    """Minimal text-to-image interface."""

    @abstractmethod
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
        """Generate one image. A fixed *seed* makes the result reproducible."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend is reachable."""
