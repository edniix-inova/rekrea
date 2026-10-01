"""
Prompt templates
================

Turns a short idea into a structured prompt for an image-generation model.
"""

import json
from dataclasses import dataclass
from typing import Optional

from .base import PromptBackend

IMAGE_PROMPT_SYSTEM = """\
You write prompts for text-to-image diffusion models.
Given a short idea, reply with a JSON object with exactly two string fields:
"prompt": one vivid paragraph (max 60 words) describing subject, setting, composition,
lighting, style and mood. No preamble, no quotes around it.
"negative_prompt": a short comma-separated list of things to avoid (e.g. blur, text,
watermark, extra limbs).
Do not add any other fields or text."""


@dataclass
class ImagePrompt:
    """A prompt (and optional negative prompt) for an image-generation model."""

    prompt: str
    negative_prompt: str = ""


def generate_image_prompt(
    idea: str,
    backend: PromptBackend,
    *,
    style: Optional[str] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
) -> ImagePrompt:
    """Expand a short *idea* into an :class:`ImagePrompt` using *backend*.

    If the model does not return valid JSON, its raw text is used as the prompt.
    """
    request = f"Idea: {idea.strip()}"
    if style:
        request += f"\nStyle: {style.strip()}"

    raw = backend.generate(
        request,
        system=IMAGE_PROMPT_SYSTEM,
        json_mode=True,
        temperature=temperature,
        seed=seed,
    )
    try:
        data = json.loads(raw)
        prompt = str(data.get("prompt", "")).strip()
        negative = str(data.get("negative_prompt", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        prompt, negative = raw.strip(), ""
    return ImagePrompt(prompt=prompt or raw.strip(), negative_prompt=negative)
