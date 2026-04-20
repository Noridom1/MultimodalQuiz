from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None  # type: ignore[assignment]

from src.document_understanding.extractor import DocumentExtractor
from src.utils.io import write_json


def main() -> int:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Test semantic extraction on an input markdown file.")
    parser.add_argument("markdown_path", type=Path, help="Path to the markdown file to extract")
    parser.add_argument(
        "--backend",
        choices=["rule", "langchain"],
        default=os.getenv("QUIZGEN_EXTRACTOR_BACKEND", "langchain"),
        help="Extractor backend (default from QUIZGEN_EXTRACTOR_BACKEND)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "mistral"],
        default=os.getenv("QUIZGEN_LLM_PROVIDER", "mistral"),
        help="LLM provider for langchain backend (default from QUIZGEN_LLM_PROVIDER)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("QUIZGEN_LLM_MODEL"),
        help="Model name override (default from QUIZGEN_LLM_MODEL)",
    )
    parser.add_argument(
        "--granularity",
        choices=["coarse", "balanced", "fine"],
        default=os.getenv("QUIZGEN_EXTRACTION_GRANULARITY", "balanced"),
        help="Extraction granularity (default from QUIZGEN_EXTRACTION_GRANULARITY)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Number of chunks per LLM request")
    parser.add_argument("--max-calls", type=int, default=24, help="Maximum number of LLM calls")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. If omitted, result is printed.",
    )
    args = parser.parse_args()

    if not args.markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {args.markdown_path}")

    markdown_text = args.markdown_path.read_text(encoding="utf-8", errors="ignore")
    extractor = DocumentExtractor(
        backend=args.backend,
        provider=args.provider,
        granularity=args.granularity,
        model=args.model,
        batch_size=args.batch_size,
        max_calls=args.max_calls,
    )
    extracted = extractor.extract(markdown_text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, {"markdown_path": str(args.markdown_path), "extracted": extracted})
        print(f"Saved extraction output to {args.output}")
    else:
        print(extracted)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
