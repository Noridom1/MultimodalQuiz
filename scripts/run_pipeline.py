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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory that will contain outputs/<run_id>/",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional explicit run ID")
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions to plan and generate")
    parser.add_argument("--easy", type=float, default=0.4, help="Easy difficulty ratio")
    parser.add_argument("--medium", type=float, default=0.4, help="Medium difficulty ratio")
    parser.add_argument("--hard", type=float, default=0.2, help="Hard difficulty ratio")
    parser.add_argument("--mock-image", action="store_true", help="Use mock image outputs instead of calling the provider")
    parser.add_argument(
        "--mock-question",
        action="store_true",
        help="Use mock question outputs instead of calling the question LLM",
    )
    args = parser.parse_args()

    pipeline = QuizGenerationPipeline()
    pipeline.run(
        args.document_path,
        output_root=args.output_root,
        run_id=args.run_id,
        num_questions=args.num_questions,
        difficulty_distribution={
            "easy": args.easy,
            "medium": args.medium,
            "hard": args.hard,
        },
        mock_image=args.mock_image,
        mock_question=args.mock_question,
    )


if __name__ == "__main__":
    main()
