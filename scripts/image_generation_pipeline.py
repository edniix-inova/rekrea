"""
Image Generation Pipeline — Local CLI
=====================================

Idea -> image prompt (Ollama) -> image (imagegen service). The prompt step can
be skipped by passing a prompt directly or a JSON file written by
prompt_generation_pipeline.py. Results are saved to
scripts/playground_data/output/images/ as a PNG plus a JSON file with the
prompt, seed and model that reproduce it.

Usage
-----
    python scripts/image_generation_pipeline.py "a vintage motorcycle at dawn"
    python scripts/image_generation_pipeline.py --prompt "a red fox in snow" --seed 7
    python scripts/image_generation_pipeline.py --prompt-file path/to/prompt.json

Requirements
------------
    No extra Python packages. Start the services first:
        docker compose -f services/docker-compose.yml up -d
        docker exec rekrea-ollama ollama pull llama3.2:3b
    (The first image request downloads the model, which takes a few minutes.)
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Allow running from the project root or from scripts/ directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rekrea.modules.image_generation import ImageServiceError, ServiceBackend
from rekrea.modules.prompting import OllamaBackend, OllamaError, generate_image_prompt

OUTPUT_DIR = Path(__file__).parent / "playground_data" / "output" / "images"


def _fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an image from an idea or a prompt.")
    parser.add_argument("idea", nargs="?", help="Short idea; expanded into a prompt with Ollama.")
    parser.add_argument("--prompt", help="Use this prompt directly (skips Ollama).")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt (with --prompt).")
    parser.add_argument("--prompt-file", type=Path,
                        help="JSON from prompt_generation_pipeline.py (skips Ollama).")
    parser.add_argument("--style", help="Style hint for the prompt step.")
    parser.add_argument("--model", help="Ollama model for the prompt step.")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, help="Seed for reproducible images.")
    args = parser.parse_args()

    sources = [bool(args.idea), bool(args.prompt), bool(args.prompt_file)]
    if sum(sources) != 1:
        return _fail("give exactly one of: an idea, --prompt, or --prompt-file.")

    idea = args.idea
    ollama_model = None

    # --- Step 1: prompt ---------------------------------------------------
    if args.prompt_file:
        data = json.loads(args.prompt_file.read_text(encoding="utf-8"))
        prompt, negative = data["prompt"], data.get("negative_prompt", "")
        idea = data.get("idea")
    elif args.prompt:
        prompt, negative = args.prompt, args.negative_prompt
    else:
        llm = OllamaBackend(model=args.model)
        if not llm.is_available():
            return _fail(
                f"Ollama model '{llm.model}' is not available at {llm.host}. "
                "See services/README.md."
            )
        try:
            result = generate_image_prompt(args.idea, llm, style=args.style, seed=args.seed)
            llm.unload()  # free VRAM for the image model
        except OllamaError as exc:
            return _fail(str(exc))
        prompt, negative, ollama_model = result.prompt, result.negative_prompt, llm.model
    print(f"Prompt:   {prompt}")
    print(f"Negative: {negative}")

    # --- Step 2: image ----------------------------------------------------
    images = ServiceBackend()
    if not images.is_available():
        return _fail(f"image service not reachable at {images.url}. See services/README.md.")
    print("Generating image (the first request loads the model, which can take a while)...")
    try:
        image = images.generate(
            prompt,
            negative_prompt=negative,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
    except ImageServiceError as exc:
        return _fail(str(exc))

    # --- Save -------------------------------------------------------------
    slug = re.sub(r"[^a-z0-9]+", "-", (idea or prompt).lower()).strip("-")[:40] or "image"
    stem = OUTPUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}"
    png = image.save(stem.with_suffix(".png"))
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "idea": idea,
                "prompt": prompt,
                "negative_prompt": negative,
                "ollama_model": ollama_model,
                "image_model": image.model,
                "seed": image.seed,
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved to: {png} (seed {image.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
