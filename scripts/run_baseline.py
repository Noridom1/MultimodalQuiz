from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import BaselineRunner


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone baseline quiz generator.")
    parser.add_argument("document_path", nargs="?", type=Path, help="Path to the source document")
    parser.add_argument(
        "--document-text",
        type=str,
        default=None,
        help="Raw document text provided directly instead of a file path",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs-baseline",
        help="Directory that will contain outputs/<run_id>/",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional explicit run ID")
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions to generate")
    parser.add_argument("--easy", type=float, default=0.4, help="Easy difficulty ratio")
    parser.add_argument("--medium", type=float, default=0.4, help="Medium difficulty ratio")
    parser.add_argument("--hard", type=float, default=0.2, help="Hard difficulty ratio")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Standard logging level for console output.",
    )
    args = parser.parse_args()

    if args.document_text is None and args.document_path is None:
        parser.error("either document_path or --document-text must be provided")

    _configure_logging(args.log_level)

    runner = BaselineRunner()
    runner.run(
        args.document_path,
        document_text=args.document_text,
        output_root=args.output_root,
        run_id=args.run_id,
        num_questions=args.num_questions,
        difficulty_distribution={
            "easy": args.easy,
            "medium": args.medium,
            "hard": args.hard,
        },
    )


if __name__ == "__main__":
    main()