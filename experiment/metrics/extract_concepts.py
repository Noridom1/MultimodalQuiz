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

from experiment.metrics.concept_extraction import (
    extract_document_concepts,
    extract_question_concepts,
    load_document_text,
)
from src.utils.io import read_json, write_json
from src.utils.llm import LLMClient

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


def _resolve_document_pdf(doc_folder: Path, raw_root: Path) -> Path | None:
    if _DOC_FOLDER_PATTERN.match(doc_folder.name):
        candidate = raw_root / f"{doc_folder.name}.pdf"
        if candidate.exists():
            return candidate
    return None


def _load_questions(doc_folder: Path) -> list[dict[str, Any]]:
    questions_path = doc_folder / "generation" / "questions.json"
    if not questions_path.exists():
        return []

    payload = read_json(questions_path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _save_concept_artifacts(
    concept_root: Path,
    *,
    doc_id: str,
    document_path: Path,
    question_count: int,
    document_concepts: list[str],
    question_concepts_payload: dict[str, Any],
) -> dict[str, str]:
    doc_dir = concept_root / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    document_concepts_path = doc_dir / "document_concepts.json"
    question_concepts_path = doc_dir / "question_concepts.json"

    write_json(
        document_concepts_path,
        {
            "doc_id": doc_id,
            "source_document": str(document_path),
            "created_at": _iso_now(),
            "concepts": document_concepts,
        },
    )
    write_json(
        question_concepts_path,
        {
            "doc_id": doc_id,
            "question_count": question_count,
            "created_at": _iso_now(),
            "all_concepts": question_concepts_payload.get("all_concepts", []),
            "per_question": question_concepts_payload.get("per_question", []),
        },
    )

    return {
        "document_concepts_path": str(document_concepts_path),
        "question_concepts_path": str(question_concepts_path),
    }


def _existing_artifacts(concept_root: Path, doc_id: str) -> dict[str, str] | None:
    doc_dir = concept_root / doc_id
    document_concepts_path = doc_dir / "document_concepts.json"
    question_concepts_path = doc_dir / "question_concepts.json"

    if document_concepts_path.exists() and question_concepts_path.exists():
        return {
            "document_concepts_path": str(document_concepts_path),
            "question_concepts_path": str(question_concepts_path),
        }
    return None


def extract_baseline_concepts(
    *,
    outputs_baseline_root: Path,
    raw_documents_root: Path,
    experiment_root: Path,
    resume: bool = True,
    max_retries: int = 5,
    initial_backoff_seconds: float = 1.0,
    max_backoff_seconds: float = 20.0,
) -> dict[str, Any]:
    llm = LLMClient()
    concept_root = experiment_root / "metrics" / "concepts"
    concept_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for doc_folder in _discover_doc_folders(outputs_baseline_root):
        doc_id = doc_folder.name

        existing = _existing_artifacts(concept_root, doc_id)
        if resume and existing is not None:
            records.append(
                {
                    "doc_id": doc_id,
                    "status": "already_completed",
                    "artifacts": existing,
                    "reason": "Skipped because concept artifacts already exist",
                }
            )
            continue

        questions = _load_questions(doc_folder)
        if not questions:
            records.append(
                {
                    "doc_id": doc_id,
                    "status": "skipped",
                    "reason": "Missing or empty generation/questions.json",
                }
            )
            continue

        document_pdf = _resolve_document_pdf(doc_folder, raw_documents_root)
        if document_pdf is None:
            records.append(
                {
                    "doc_id": doc_id,
                    "status": "skipped",
                    "reason": "Could not map output folder to data/raw/docX.pdf",
                }
            )
            continue

        document_text = load_document_text(document_pdf)
        try:
            document_concepts = extract_document_concepts(
                document_text,
                llm,
                max_retries=max_retries,
                initial_backoff_seconds=initial_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
            )
            question_concepts_payload = extract_question_concepts(
                questions,
                llm,
                max_retries=max_retries,
                initial_backoff_seconds=initial_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
            )
        except Exception as exc:
            records.append(
                {
                    "doc_id": doc_id,
                    "status": "failed",
                    "reason": str(exc),
                }
            )
            continue

        artifact_paths = _save_concept_artifacts(
            concept_root,
            doc_id=doc_id,
            document_path=document_pdf,
            question_count=len(questions),
            document_concepts=document_concepts,
            question_concepts_payload=question_concepts_payload,
        )

        records.append(
            {
                "doc_id": doc_id,
                "status": "completed",
                "artifacts": artifact_paths,
                "document_concepts": len(document_concepts),
                "question_concepts": len(question_concepts_payload.get("all_concepts", [])),
            }
        )

    completed = [record for record in records if record.get("status") == "completed"]
    already_completed = [record for record in records if record.get("status") == "already_completed"]
    failed = [record for record in records if record.get("status") == "failed"]
    summary = {
        "created_at": _iso_now(),
        "outputs_baseline_root": str(outputs_baseline_root),
        "raw_documents_root": str(raw_documents_root),
        "concept_root": str(concept_root),
        "resume": resume,
        "retry": {
            "max_retries": max_retries,
            "initial_backoff_seconds": initial_backoff_seconds,
            "max_backoff_seconds": max_backoff_seconds,
        },
        "documents_total": len(records),
        "documents_completed": len(completed),
        "documents_already_completed": len(already_completed),
        "documents_failed": len(failed),
        "documents_skipped": len(records) - len(completed) - len(already_completed) - len(failed),
        "records": records,
    }

    summary_path = concept_root / "concept_extraction_summary.json"
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract concepts from baseline source documents and generated questions."
    )
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
        help="Root folder containing docX.pdf files",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "experiment",
        help="Experiment folder used to store extracted concept artifacts",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable continue mode and re-run all documents",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retries for transient LLM errors",
    )
    parser.add_argument(
        "--initial-backoff-seconds",
        type=float,
        default=1.0,
        help="Initial retry backoff in seconds",
    )
    parser.add_argument(
        "--max-backoff-seconds",
        type=float,
        default=20.0,
        help="Maximum retry backoff in seconds",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    summary = extract_baseline_concepts(
        outputs_baseline_root=args.outputs_baseline_root,
        raw_documents_root=args.raw_documents_root,
        experiment_root=args.experiment_root,
        resume=not args.no_resume,
        max_retries=max(0, args.max_retries),
        initial_backoff_seconds=max(0.0, args.initial_backoff_seconds),
        max_backoff_seconds=max(0.0, args.max_backoff_seconds),
    )
    print(f"Completed concept extraction for {summary['documents_completed']} documents")
    print(f"Summary written to: {summary['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
