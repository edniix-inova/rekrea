"""
Prompting backend interface
===========================

A prompting backend turns a text prompt into generated text. Keeping the
interface this small lets callers stay independent of the engine (Ollama, a
hosted API, an in-process model...). Add a new engine by subclassing
:class:`PromptBackend`.
"""

from abc import ABC, abstractmethod
from typing import Optional


class PromptBackend(ABC):
    """Minimal text-generation interface."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Return the generated text for *prompt*.

        Parameters
        ----------
        prompt:
            User prompt.
        system:
            Optional system instruction.
        json_mode:
            Ask the engine to return valid JSON.
        temperature:
            Sampling temperature. ``None`` keeps the engine default.
        seed:
            Fixed seed for reproducible output where the engine supports it.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the engine is reachable and the model is usable."""
