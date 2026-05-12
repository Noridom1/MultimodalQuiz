from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.metrics.extract_concepts import extract_baseline_concepts
from experiment.metrics.metrics_calculator import compute_metrics
from src.utils.io import read_json, write_json

_DOC_FOLDER_PATTERN = re.compile(r"^doc\d+$", re.IGNORECASE)


def _iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _discover_doc_folders(outputs_baseline_root: Path) -> list[Path]:
    if not outputs_baseline_root.exists():
        return []

    return sorted(
        [
            child
            for child in outputs_baseline_root.iterdir()
            if child.is_dir() and _DOC_FOLDER_PATTERN.match(child.name)
        ],
        key=lambda p: p.name,
    )


def _load_concept_payload(concept_root: Path, doc_id: str) -> tuple[list[str], list[str], dict[str, str]] | None:
    doc_dir = concept_root / doc_id
    document_concepts_path = doc_dir / "document_concepts.json"
    question_concepts_path = doc_dir / "question_concepts.json"

    if not document_concepts_path.exists() or not question_concepts_path.exists():
        return None

    document_payload = read_json(document_concepts_path)
    question_payload = read_json(question_concepts_path)

    if not isinstance(document_payload, dict) or not isinstance(question_payload, dict):
        return None

    document_concepts_obj = document_payload.get("concepts", [])
    quiz_concepts_obj = question_payload.get("all_concepts", [])

    if not isinstance(document_concepts_obj, list) or not isinstance(quiz_concepts_obj, list):
        return None

    document_concepts = [str(item).strip() for item in document_concepts_obj if str(item).strip()]
    quiz_concepts = [str(item).strip() for item in quiz_concepts_obj if str(item).strip()]

    return (
        document_concepts,
        quiz_concepts,
        {
            "document_concepts_path": str(document_concepts_path),
            "question_concepts_path": str(question_concepts_path),
        },
    )


def evaluate_baseline_outputs(
    *,
    outputs_baseline_root: Path,
    raw_documents_root: Path,
    experiment_root: Path,
    coverage_threshold: float = 0.75,
    auto_extract: bool = False,
    resume: bool = True,
    max_retries: int = 5,
    initial_backoff_seconds: float = 1.0,
    max_backoff_seconds: float = 20.0,
) -> dict[str, Any]:
    metrics_root = experiment_root / "metrics"
    concept_root = metrics_root / "concepts"
    results_root = metrics_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    if auto_extract:
        extract_baseline_concepts(
            outputs_baseline_root=outputs_baseline_root,
            raw_documents_root=raw_documents_root,
            experiment_root=experiment_root,
            resume=resume,
            max_retries=max_retries,
            initial_backoff_seconds=initial_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
        )

    records: list[dict[str, Any]] = []

    for doc_folder in _discover_doc_folders(outputs_baseline_root):
        doc_id = doc_folder.name
        concept_payload = _load_concept_payload(concept_root, doc_id)
        if concept_payload is None:
            records.append(
                {
                    "doc_id": doc_id,
                    "status": "skipped",
                    "reason": "Missing concept files. Run extract_concepts first or pass --auto-extract.",
                }
            )
            continue

        document_concepts, quiz_concepts, concept_paths = concept_payload

        metric_payload = compute_metrics(
            document_concepts,
            quiz_concepts,
            coverage_threshold=coverage_threshold,
        )

        per_doc_result_path = results_root / f"{doc_id}_metrics.json"
        per_doc_payload = {
            "doc_id": doc_id,
            "created_at": _iso_now(),
            "coverage_threshold": coverage_threshold,
            "source_questions": str(doc_folder / "generation" / "questions.json"),
            "metrics": metric_payload,
            "artifacts": concept_paths,
        }
        write_json(per_doc_result_path, per_doc_payload)

        records.append(
            {
                "doc_id": doc_id,
                "status": "completed",
                "result_path": str(per_doc_result_path),
                "metrics": metric_payload,
            }
        )

    completed = [record for record in records if record.get("status") == "completed"]

    summary = {
        "created_at": _iso_now(),
        "outputs_baseline_root": str(outputs_baseline_root),
        "coverage_threshold": coverage_threshold,
        "auto_extract": auto_extract,
        "resume": resume,
        "retry": {
            "max_retries": max_retries,
            "initial_backoff_seconds": initial_backoff_seconds,
            "max_backoff_seconds": max_backoff_seconds,
        },
        "documents_total": len(records),
        "documents_completed": len(completed),
        "documents_skipped": len(records) - len(completed),
        "records": records,
    }

    summary_path = results_root / "baseline_metrics_summary.json"
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute baseline concept metrics for quiz outputs.")
    parser.add_argument(
        "--outputs-baseline-root",
        type=Path,
        default=PROJECT_ROOT / "outputs-baseline",
        help="Root folder containing docX baseline output folders",
    )
    parser.add_argument(
        "--raw-documents-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Root folder containing docX.pdf files (used when --auto-extract is enabled)",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "experiment",
        help="Experiment folder used to store concepts and metric artifacts",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for thresholded coverage",
    )
    parser.add_argument(
        "--auto-extract",
        action="store_true",
        help="Run concept extraction before metric computation",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable continue mode and re-run all documents when --auto-extract is used",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retries for transient LLM errors during auto-extract",
    )
    parser.add_argument(
        "--initial-backoff-seconds",
        type=float,
        default=1.0,
        help="Initial retry backoff in seconds during auto-extract",
    )
    parser.add_argument(
        "--max-backoff-seconds",
        type=float,
        default=20.0,
        help="Maximum retry backoff in seconds during auto-extract",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    summary = evaluate_baseline_outputs(
        outputs_baseline_root=args.outputs_baseline_root,
        raw_documents_root=args.raw_documents_root,
        experiment_root=args.experiment_root,
        coverage_threshold=args.coverage_threshold,
        auto_extract=args.auto_extract,
        resume=not args.no_resume,
        max_retries=max(0, args.max_retries),
        initial_backoff_seconds=max(0.0, args.initial_backoff_seconds),
        max_backoff_seconds=max(0.0, args.max_backoff_seconds),
    )
    print(f"Completed metrics for {summary['documents_completed']} documents")
    print(f"Summary written to: {summary['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
