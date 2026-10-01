"""
Prompt Generation Pipeline — Local CLI
======================================

Expands a short idea into a structured image-generation prompt using a local
Ollama server (see services/README.md) and saves it as JSON to
scripts/playground_data/output/prompts/.

Usage
-----
    python scripts/prompt_generation_pipeline.py "a vintage motorcycle at dawn"
    python scripts/prompt_generation_pipeline.py "a lighthouse" --style "watercolor" --seed 42
    python scripts/prompt_generation_pipeline.py "a lighthouse" --model qwen2.5:3b

Requirements
------------
    No extra Python packages. An Ollama server must be running with the model pulled:
        docker compose -f services/docker-compose.yml up -d
        docker exec rekrea-ollama ollama pull llama3.2:3b
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

from rekrea.modules.prompting import OllamaBackend, OllamaError, generate_image_prompt

OUTPUT_DIR = Path(__file__).parent / "playground_data" / "output" / "prompts"


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand an idea into an image prompt.")
    parser.add_argument("idea", help="Short description of the image you want.")
    parser.add_argument("--style", help="Optional style hint, e.g. 'watercolor'.")
    parser.add_argument("--model", help="Ollama model (default: REKREA_OLLAMA_MODEL or llama3.2:3b).")
    parser.add_argument("--host", help="Ollama URL (default: OLLAMA_HOST or http://localhost:11434).")
    parser.add_argument("--temperature", type=float, help="Sampling temperature.")
    parser.add_argument("--seed", type=int, help="Seed for reproducible output.")
    args = parser.parse_args()

    backend = OllamaBackend(model=args.model, host=args.host)
    if not backend.is_available():
        print(
            f"Ollama model '{backend.model}' is not available at {backend.host}.\n"
            f"Start the server and pull it:\n"
            f"  docker compose -f services/docker-compose.yml up -d\n"
            f"  docker exec rekrea-ollama ollama pull {backend.model}",
            file=sys.stderr,
        )
        return 1

    try:
        result = generate_image_prompt(
            args.idea,
            backend,
            style=args.style,
            temperature=args.temperature,
            seed=args.seed,
        )
    except OllamaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", args.idea.lower()).strip("-")[:40] or "prompt"
    out_path = OUTPUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.json"
    out_path.write_text(
        json.dumps(
            {
                "idea": args.idea,
                "style": args.style,
                "model": backend.model,
                "seed": args.seed,
                "prompt": result.prompt,
                "negative_prompt": result.negative_prompt,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Prompt:   {result.prompt}")
    print(f"Negative: {result.negative_prompt}")
    print(f"Saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
