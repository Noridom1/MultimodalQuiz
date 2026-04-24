from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports like `from src...` work
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.generator.orchestrator import GenerationOrchestrator


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Orchestrate image + question generation from a quiz plan")
    parser.add_argument(
        "--plan-json",
        type=Path,
        required=True,
        help="Quiz plan JSON file used to build prompts and generate outputs",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--prompts-json", type=Path, default=None, help="Combined prompts JSON file (legacy override)")
    group.add_argument(
        "--prompts-dir",
        type=Path,
        default=None,
        help="Directory containing question_prompt.json and image_prompts.json (legacy override)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write generated questions JSON. Defaults to data/questions/<run_id>.json",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional explicit run ID for the question set")
    parser.add_argument(
        "--image-paths",
        nargs="*",
        default=None,
        help="Optional source image paths for the image generator",
    )
    parser.add_argument("--mock-image", action="store_true", help="Use mock image outputs instead of calling provider")
    parser.add_argument("--mock-question", action="store_true", help="Use mock question outputs instead of calling LLM")

    args = parser.parse_args(argv)

    orchestrator = GenerationOrchestrator()
    prompts = None
    if args.prompts_dir is not None:
        prompts = orchestrator.load_prompts_from_dir(args.prompts_dir)
    elif args.prompts_json is not None:
        prompts = json.loads(args.prompts_json.read_text(encoding="utf-8"))

    orchestrator.run(
        args.plan_json,
        prompts=prompts,
        output_path=args.output,
        run_id=args.run_id,
        image_paths=args.image_paths,
        mock_image=args.mock_image,
        mock_question=args.mock_question,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
