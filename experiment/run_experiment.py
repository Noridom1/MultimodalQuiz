from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import BaselineRunner
from src.pipeline import QuizGenerationPipeline
from src.utils.io import relative_path, write_json


def _iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"\s+", "_", name.strip())
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "document"


def _discover_pdf_files(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        candidates = [path for path in input_dir.rglob("*") if path.is_file()]
    else:
        candidates = [path for path in input_dir.glob("*") if path.is_file()]
    pdfs = [path for path in candidates if path.suffix.lower() == ".pdf"]
    return sorted(pdfs)


def _resolve_run_id(
    document_stem: str,
    *,
    num_questions: int,
    output_root: Path,
    used_run_ids: set[str],
) -> str:
    base = f"{_sanitize_name(document_stem)}_{num_questions}"
    candidate = base
    counter = 2

    while candidate in used_run_ids or (output_root / candidate).exists():
        candidate = f"{base}_{counter}"
        counter += 1

    used_run_ids.add(candidate)
    return candidate


def _configure_logger(log_path: Path, level_name: str) -> logging.Logger:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger = logging.getLogger("experiment_runner")
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def _log_event(
    logger: logging.Logger,
    *,
    event: str,
    method: str,
    document: str,
    run_id: str,
    status: str,
    elapsed_sec: float | None,
    message: str,
    level: int = logging.INFO,
) -> None:
    elapsed_text = "-" if elapsed_sec is None else f"{elapsed_sec:.3f}"
    logger.log(
        level,
        (
            f"event={event} method={method} document={document} run_id={run_id} "
            f"status={status} elapsed_sec={elapsed_text} message={message}"
        ),
    )


def _tail_file(path: Path, max_lines: int = 20) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) <= max_lines:
        return lines
    return lines[-max_lines:]


def _run_single_document(
    *,
    method: str,
    document_path: Path,
    output_root: Path,
    run_id: str,
    num_questions: int,
    difficulty_distribution: dict[str, float],
    generation_mode: str,
    mock_image: bool,
    mock_question: bool,
) -> dict[str, Any]:
    if method == "baseline":
        runner = BaselineRunner()
        return runner.run(
            document_path,
            output_root=output_root,
            run_id=run_id,
            num_questions=num_questions,
            difficulty_distribution=difficulty_distribution,
        )

    pipeline = QuizGenerationPipeline()
    return pipeline.run(
        document_path,
        output_root=output_root,
        run_id=run_id,
        num_questions=num_questions,
        difficulty_distribution=difficulty_distribution,
        generation_mode=generation_mode,
        mock_image=mock_image,
        mock_question=mock_question,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch experiment runner for baseline and mmquiz methods.")
    parser.add_argument("--method", required=True, choices=("baseline", "mmquiz"), help="Method to run")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing PDF files")
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions per document")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "experiment", help="Experiment root directory")
    parser.add_argument("--easy", type=float, default=0.4, help="Easy difficulty ratio")
    parser.add_argument("--medium", type=float, default=0.4, help="Medium difficulty ratio")
    parser.add_argument("--hard", type=float, default=0.2, help="Hard difficulty ratio")
    parser.add_argument("--log-level", type=str, default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    parser.add_argument("--recursive", action="store_true", help="Recursively discover PDF files")
    parser.add_argument("--limit", type=int, default=None, help="Run at most N discovered PDFs")
    parser.add_argument("--dry-run", action="store_true", help="Show planned runs without executing")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop batch execution on first failure")
    parser.add_argument("--echo-run-logs", action="store_true", help="Echo tail of each run log after completion")
    parser.add_argument(
        "--generation-mode",
        type=str,
        default="topic_agentic",
        choices=("topic_agentic", "legacy"),
        help="MMQuiz generation mode (ignored for baseline)",
    )
    parser.add_argument("--mock-image", action="store_true", help="MMQuiz only: mock image outputs")
    parser.add_argument("--mock-question", action="store_true", help="MMQuiz only: mock question outputs")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.num_questions <= 0:
        parser.error("--num-questions must be greater than zero")

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        parser.error(f"--input-dir is not a valid directory: {args.input_dir}")

    method_root = args.output_root / args.method
    method_root.mkdir(parents=True, exist_ok=True)

    logger = _configure_logger(method_root / "experiment.log", args.log_level)

    discovered = _discover_pdf_files(args.input_dir, recursive=args.recursive)
    if args.limit is not None:
        discovered = discovered[: max(0, args.limit)]

    started_at = _iso_now()
    batch_timer = time.perf_counter()

    _log_event(
        logger,
        event="experiment_started",
        method=args.method,
        document="-",
        run_id="-",
        status="started",
        elapsed_sec=None,
        message=f"input_dir={args.input_dir} files={len(discovered)} dry_run={args.dry_run}",
    )

    if not discovered:
        summary = {
            "method": args.method,
            "input_dir": str(args.input_dir),
            "num_questions": args.num_questions,
            "total_files": 0,
            "succeeded": 0,
            "failed": 0,
            "started_at": started_at,
            "finished_at": _iso_now(),
            "total_elapsed_seconds": 0.0,
            "records": [],
        }
        write_json(method_root / "experiment_summary.json", summary)
        _log_event(
            logger,
            event="experiment_finished",
            method=args.method,
            document="-",
            run_id="-",
            status="completed",
            elapsed_sec=0.0,
            message="No PDF files found",
        )
        return 0

    used_run_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0

    difficulty_distribution = {
        "easy": args.easy,
        "medium": args.medium,
        "hard": args.hard,
    }

    total = len(discovered)

    for index, document_path in enumerate(discovered, start=1):
        run_id = _resolve_run_id(
            document_path.stem,
            num_questions=args.num_questions,
            output_root=method_root,
            used_run_ids=used_run_ids,
        )
        run_started_at = _iso_now()
        file_timer = time.perf_counter()

        print(f"[{index}/{total}] START {document_path.name} -> {run_id}")
        _log_event(
            logger,
            event="document_started",
            method=args.method,
            document=str(document_path),
            run_id=run_id,
            status="started",
            elapsed_sec=None,
            message=f"queued_index={index} of {total}",
        )

        if args.dry_run:
            elapsed = time.perf_counter() - file_timer
            records.append(
                {
                    "document_path": str(document_path),
                    "document_name": document_path.name,
                    "run_id": run_id,
                    "method": args.method,
                    "status": "dry_run",
                    "started_at": run_started_at,
                    "finished_at": _iso_now(),
                    "elapsed_seconds": round(elapsed, 6),
                    "output_root": str(method_root / run_id),
                    "manifest_path": None,
                    "log_path": None,
                    "error_type": None,
                    "error_message": None,
                }
            )
            _log_event(
                logger,
                event="document_dry_run",
                method=args.method,
                document=str(document_path),
                run_id=run_id,
                status="dry_run",
                elapsed_sec=elapsed,
                message="Skipped execution due to --dry-run",
            )
            continue

        try:
            result = _run_single_document(
                method=args.method,
                document_path=document_path,
                output_root=method_root,
                run_id=run_id,
                num_questions=args.num_questions,
                difficulty_distribution=difficulty_distribution,
                generation_mode=args.generation_mode,
                mock_image=args.mock_image,
                mock_question=args.mock_question,
            )

            elapsed = time.perf_counter() - file_timer
            run_root = Path(result.get("run_root", method_root / run_id))
            manifest_path = Path(result.get("manifest", run_root / "manifest.json"))
            log_path = Path(result.get("log_path", run_root / "logs" / "pipeline.log"))

            succeeded += 1
            records.append(
                {
                    "document_path": str(document_path),
                    "document_name": document_path.name,
                    "run_id": run_id,
                    "method": args.method,
                    "status": "success",
                    "started_at": run_started_at,
                    "finished_at": _iso_now(),
                    "elapsed_seconds": round(elapsed, 6),
                    "output_root": str(run_root),
                    "manifest_path": str(manifest_path),
                    "log_path": str(log_path),
                    "error_type": None,
                    "error_message": None,
                }
            )

            print(f"[{index}/{total}] OK    {document_path.name} ({elapsed:.2f}s)")
            _log_event(
                logger,
                event="document_completed",
                method=args.method,
                document=str(document_path),
                run_id=run_id,
                status="success",
                elapsed_sec=elapsed,
                message=f"run_root={relative_path(run_root, PROJECT_ROOT)}",
            )

            if args.echo_run_logs and log_path.exists():
                logger.info("event=document_log_tail method=%s document=%s run_id=%s status=info elapsed_sec=- message=tail_begin", args.method, document_path, run_id)
                for line in _tail_file(log_path, max_lines=20):
                    logger.info("[run-log:%s] %s", run_id, line)
                logger.info("event=document_log_tail method=%s document=%s run_id=%s status=info elapsed_sec=- message=tail_end", args.method, document_path, run_id)

        except Exception as exc:
            elapsed = time.perf_counter() - file_timer
            failed += 1
            records.append(
                {
                    "document_path": str(document_path),
                    "document_name": document_path.name,
                    "run_id": run_id,
                    "method": args.method,
                    "status": "failed",
                    "started_at": run_started_at,
                    "finished_at": _iso_now(),
                    "elapsed_seconds": round(elapsed, 6),
                    "output_root": str(method_root / run_id),
                    "manifest_path": None,
                    "log_path": None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

            print(f"[{index}/{total}] FAIL  {document_path.name} {exc}")
            _log_event(
                logger,
                event="document_failed",
                method=args.method,
                document=str(document_path),
                run_id=run_id,
                status="failed",
                elapsed_sec=elapsed,
                message=f"{type(exc).__name__}: {exc}",
                level=logging.ERROR,
            )

            if args.stop_on_error:
                _log_event(
                    logger,
                    event="experiment_stopped",
                    method=args.method,
                    document=str(document_path),
                    run_id=run_id,
                    status="stopped",
                    elapsed_sec=elapsed,
                    message="Stopping due to --stop-on-error",
                    level=logging.ERROR,
                )
                break

    total_elapsed = time.perf_counter() - batch_timer
    finished_at = _iso_now()

    summary = {
        "method": args.method,
        "input_dir": str(args.input_dir),
        "num_questions": args.num_questions,
        "difficulty_distribution": difficulty_distribution,
        "generation_mode": args.generation_mode if args.method == "mmquiz" else None,
        "total_files": len(discovered),
        "succeeded": succeeded,
        "failed": failed,
        "started_at": started_at,
        "finished_at": finished_at,
        "total_elapsed_seconds": round(total_elapsed, 6),
        "records": records,
    }

    summary_path = method_root / "experiment_summary.json"
    write_json(summary_path, summary)

    _log_event(
        logger,
        event="experiment_finished",
        method=args.method,
        document="-",
        run_id="-",
        status="completed" if failed == 0 else "completed_with_failures",
        elapsed_sec=total_elapsed,
        message=f"succeeded={succeeded} failed={failed} summary={relative_path(summary_path, PROJECT_ROOT)}",
    )

    print(
        f"SUMMARY method={args.method} succeeded={succeeded} failed={failed} "
        f"total={len(discovered)} elapsed={total_elapsed:.2f}s"
    )

    if args.dry_run:
        return 0
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
