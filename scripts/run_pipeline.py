from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import QuizGenerationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multimodal quiz generation pipeline.")
    parser.add_argument("document_path", type=Path, help="Path to the source document")
    parser.add_argument("output_path", type=Path, help="Path to write generated quiz output")
    args = parser.parse_args()

    pipeline = QuizGenerationPipeline()
    pipeline.run(args.document_path, output_path=args.output_path)


if __name__ == "__main__":
    main()
