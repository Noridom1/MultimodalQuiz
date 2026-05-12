from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None  # type: ignore[assignment]

from src.baseline.baseline_prompt import build_baseline_prompt
from src.knowledge.schema import Question
from src.utils.io import append_jsonl, relative_path, write_json
from src.utils.llm import LLMClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_DIFFICULTY_DISTRIBUTION = {
    "easy": 0.4,
    "medium": 0.4,
    "hard": 0.2,
}


@dataclass(frozen=True)
class BaselineRunContext:
    output_root: Path
    run_id: str
    run_root: Path
    document_dir: Path
    generation_dir: Path
    logs_dir: Path
    manifest_path: Path
    log_path: Path

    @classmethod
    def create(
        cls,
        source_label: str,
        *,
        output_root: str | Path | None = None,
        run_id: str | None = None,
    ) -> "BaselineRunContext":
        root = Path(output_root) if output_root is not None else PROJECT_ROOT / "outputs"
        effective_run_id = run_id or _generate_run_id(source_label)
        run_root = root / effective_run_id
        document_dir = run_root / "document"
        generation_dir = run_root / "generation"
        logs_dir = run_root / "logs"

        for path in (document_dir, generation_dir, logs_dir):
            path.mkdir(parents=True, exist_ok=True)

        return cls(
            output_root=root,
            run_id=effective_run_id,
            run_root=run_root,
            document_dir=document_dir,
            generation_dir=generation_dir,
            logs_dir=logs_dir,
            manifest_path=run_root / "manifest.json",
            log_path=logs_dir / "pipeline.log",
        )


def _generate_run_id(source_label: str) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", source_label).strip("-") or "document"
    return f"{timestamp}_{slug}_{uuid4().hex[:6]}"


def _log_event(context: BaselineRunContext, stage: str, event: str, message: str, **details: object) -> None:
    append_jsonl(
        context.log_path,
        {
            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            "run_id": context.run_id,
            "stage": stage,
            "event": event,
            "message": message,
            "details": details,
        },
    )


def _question_to_dict(question: Question) -> dict[str, object]:
    if hasattr(question, "model_dump"):
        return question.model_dump()
    if hasattr(question, "dict"):
        return question.dict()
    raise TypeError("Unsupported question type for serialization")


class BaselineRunner:
    def __init__(self, *, llm_client: LLMClient | None = None, max_retries: int = 2) -> None:
        self._llm_client = llm_client or LLMClient()
        self._max_retries = max_retries

    def run(
        self,
        document_path: str | Path | None = None,
        *,
        document_text: str | None = None,
        output_root: str | Path | None = None,
        run_id: str | None = None,
        num_questions: int = 5,
        difficulty_distribution: dict[str, float] | None = None,
    ) -> dict[str, object]:
        if num_questions <= 0:
            raise ValueError("num_questions must be greater than zero.")

        effective_distribution = difficulty_distribution or DEFAULT_DIFFICULTY_DISTRIBUTION
        self._validate_difficulty_distribution(effective_distribution)

        source_label, document_text = self._load_document(document_path, document_text=document_text)
        context = BaselineRunContext.create(source_label, output_root=output_root, run_id=run_id)

        _log_event(
            context,
            "baseline",
            "started",
            "Baseline generation started",
            source_label=source_label,
            output_root=str(context.output_root),
        )

        raw_document_path = context.document_dir / "document_text.txt"
        raw_document_path.write_text(document_text, encoding="utf-8")

        prompt = build_baseline_prompt(
            document_text,
            num_questions=num_questions,
            difficulty_distribution=effective_distribution,
        )

        questions = self._generate_questions(
            prompt,
            context=context,
            run_id=context.run_id,
            source_label=source_label,
            num_questions=num_questions,
        )

        image_output_dir = context.generation_dir / "images"
        image_output_dir.mkdir(parents=True, exist_ok=True)

        question_records: list[dict[str, object]] = []
        image_artifacts: list[dict[str, object]] = []
        serialized_questions: list[dict[str, object]] = []

        for index, question in enumerate(questions, start=1):
            mock_image_ref = f"mock://baseline/{context.run_id}/{index}.png"
            question.metadata = {
                **question.metadata,
                "method": "baseline_direct_llm",
                "provider": self._provider_name(),
                "model": self._model_name(),
                "run_id": context.run_id,
                "source_label": source_label,
                "question_index": index,
            }
            question.associated_image = mock_image_ref
            question.image_grounded = False
            question.validate()

            question_dict = _question_to_dict(question)
            serialized_questions.append(question_dict)

            question_records.append(
                {
                    "index": index,
                    "question": question_dict,
                    "image_url": mock_image_ref,
                    "question_prompt": prompt,
                }
            )
            image_artifacts.append(
                {
                    "index": index,
                    "status": "mock",
                    "image_prompt": None,
                    "source_url": mock_image_ref,
                    "local_path": None,
                    "image_ref": mock_image_ref,
                }
            )

        questions_path = context.generation_dir / "questions.json"
        quiz_package_path = context.generation_dir / "quiz_package.json"
        image_artifacts_path = context.generation_dir / "image_artifacts.json"

        write_json(questions_path, serialized_questions)
        write_json(
            quiz_package_path,
            {
                "run_id": context.run_id,
                "image_dir": relative_path(image_output_dir, context.run_root),
                "results": question_records,
                "method": "baseline_direct_llm",
            },
        )
        write_json(image_artifacts_path, image_artifacts)

        manifest = {
            "run_id": context.run_id,
            "created_at": dt.datetime.utcnow().isoformat() + "Z",
            "source_document": str(document_path) if document_path is not None else "<inline-text>",
            "output_root": relative_path(context.run_root, PROJECT_ROOT),
            "stages": {
                "baseline": "completed",
            },
            "artifacts": {
                "document_text": relative_path(raw_document_path, PROJECT_ROOT),
                "questions": relative_path(questions_path, PROJECT_ROOT),
                "quiz_package": relative_path(quiz_package_path, PROJECT_ROOT),
                "image_artifacts": relative_path(image_artifacts_path, PROJECT_ROOT),
                "pipeline_log": relative_path(context.log_path, PROJECT_ROOT),
            },
            "config": {
                "method": "baseline_direct_llm",
                "provider": self._provider_name(),
                "model": self._model_name(),
                "num_questions": num_questions,
                "difficulty_distribution": effective_distribution,
            },
        }
        write_json(context.manifest_path, manifest)

        _log_event(
            context,
            "baseline",
            "completed",
            "Baseline generation completed",
            question_count=len(serialized_questions),
            manifest=str(context.manifest_path),
        )

        return {
            "run_id": context.run_id,
            "run_root": context.run_root,
            "manifest": context.manifest_path,
            "log_path": context.log_path,
            "artifacts": {
                "document_text": raw_document_path,
                "questions": questions_path,
                "quiz_package": quiz_package_path,
                "image_artifacts": image_artifacts_path,
                "image_dir": image_output_dir,
            },
            "questions": serialized_questions,
            "question_records": question_records,
            "image_artifacts": image_artifacts,
        }

    def _generate_questions(
        self,
        prompt: str,
        *,
        context: BaselineRunContext,
        run_id: str,
        source_label: str,
        num_questions: int,
    ) -> list[Question]:
        last_error: Exception | None = None
        previous_output: str | None = None

        for attempt in range(self._max_retries + 1):
            attempt_prompt = prompt
            if attempt > 0 and previous_output is not None and last_error is not None:
                attempt_prompt = self._repair_prompt(
                    original_prompt=prompt,
                    invalid_payload_text=previous_output,
                    validation_error=str(last_error),
                    num_questions=num_questions,
                )

            try:
                _log_event(
                    context,
                    "baseline",
                    "prompt_attempt",
                    "Sending prompt to LLM",
                    attempt=attempt + 1,
                    source_label=source_label,
                    prompt_length=len(attempt_prompt),
                )
                raw_output = self._llm_client.complete(attempt_prompt, system_prompt=self._system_prompt())
                previous_output = raw_output
                payload = self._parse_json_payload(raw_output)
                questions = self._parse_questions(payload, num_questions=num_questions)
                return questions
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Baseline question generation attempt %d/%d failed source=%s error=%s",
                    attempt + 1,
                    self._max_retries + 1,
                    source_label,
                    exc,
                )

        raise RuntimeError("Failed to generate a valid baseline question set after retries.") from last_error

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are an expert quiz author. Return strict JSON only. "
            "Do not output markdown, explanations outside JSON, or code fences. "
            "Generate exactly the requested number of multiple-choice questions."
        )

    @staticmethod
    def _repair_prompt(
        *,
        original_prompt: str,
        invalid_payload_text: str,
        validation_error: str,
        num_questions: int,
    ) -> str:
        return (
            f"{original_prompt}\n\n"
            "Your previous output was invalid. Return corrected JSON only. "
            f"Generate exactly {num_questions} questions. "
            f"Validation error: {validation_error}\n\n"
            f"Previous invalid JSON:\n{invalid_payload_text}"
        )

    @staticmethod
    def _parse_json_payload(raw_text: str) -> dict[str, object]:
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()

        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("LLM output must be a JSON object.")
        return parsed

    @staticmethod
    def _normalize_question_type(value: object) -> str:
        text = str(value or "multiple_choice").strip().lower().replace("-", "_")
        if text in {"mcq", "multiplechoice"}:
            return "multiple_choice"
        if text not in {"multiple_choice", "multiple_choice_question", "multiple_choice_questions"}:
            return "multiple_choice"
        return "multiple_choice"

    @staticmethod
    def _normalize_difficulty(value: object) -> str:
        text = str(value or "medium").strip().lower()
        if text not in {"easy", "medium", "hard"}:
            return "medium"
        return text

    @staticmethod
    def _parse_questions(payload: dict[str, object], *, num_questions: int) -> list[Question]:
        rows = payload.get("questions")
        if not isinstance(rows, list):
            raise RuntimeError("Baseline output must include a 'questions' list.")
        if len(rows) != num_questions:
            raise RuntimeError(f"Expected {num_questions} questions, got {len(rows)}")

        questions: list[Question] = []
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise RuntimeError(f"Question {index} must be an object.")

            options_raw = row.get("options", [])
            options = [str(item).strip() for item in options_raw] if isinstance(options_raw, list) else []
            options = [item for item in options if item]

            question = Question(
                id=f"q_{uuid4().hex[:10]}",
                question_text=str(row.get("question_text", "")).strip(),
                options=options,
                correct_answer=str(row.get("correct_answer", "")).strip(),
                explanation=str(row.get("explanation", "")).strip(),
                target_concept=str(row.get("target_concept", "")).strip() or f"concept_{index}",
                difficulty=BaselineRunner._normalize_difficulty(row.get("difficulty")),
                question_type=BaselineRunner._normalize_question_type(row.get("question_type")),
                associated_image=f"mock://baseline/{index}.png",
                image_grounded=bool(row.get("image_grounded", False)),
                metadata={
                    "method": "baseline_direct_llm",
                    "baseline_index": index,
                },
            )
            question.validate()
            questions.append(question)

        return questions

    @staticmethod
    def _validate_difficulty_distribution(dist: dict[str, float]) -> None:
        if not isinstance(dist, dict):
            raise ValueError("difficulty_distribution must be a dict")
        if not dist:
            raise ValueError("difficulty_distribution cannot be empty")

        total = 0.0
        for difficulty, proportion in dist.items():
            if difficulty not in {"easy", "medium", "hard"}:
                raise ValueError(f"Invalid difficulty: {difficulty}")
            if not isinstance(proportion, (int, float)) or proportion < 0:
                raise ValueError(f"Proportion for {difficulty} must be a positive number")
            total += float(proportion)

        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Difficulty distribution must sum to 1.0 (got {total})")

    @staticmethod
    def _provider_name() -> str:
        return os.getenv("QUIZGEN_LLM_PROVIDER", "openai")

    @staticmethod
    def _model_name() -> str | None:
        return os.getenv("QUIZGEN_LLM_MODEL")

    @staticmethod
    def _load_document(
        document_path: str | Path | None,
        *,
        document_text: str | None = None,
    ) -> tuple[str, str]:
        if document_text is not None:
            text = document_text.strip()
            if not text:
                raise ValueError("document_text cannot be empty.")
            return "inline_document", text

        if document_path is None:
            raise ValueError("Either document_path or document_text must be provided.")

        path = Path(document_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = BaselineRunner._read_pdf_text(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")

        text = text.strip()
        if not text:
            raise ValueError(f"Document is empty: {path}")

        return path.stem, text

    @staticmethod
    def _read_pdf_text(path: Path) -> str:
        try:
            from pypdf import PdfReader
        except Exception:
            try:
                from PyPDF2 import PdfReader  # type: ignore[import-not-found]
            except Exception:
                try:
                    import fitz  # type: ignore
                except Exception as exc:
                    raise RuntimeError(
                        "PDF input requires pypdf, PyPDF2, or PyMuPDF (fitz) to be installed."
                    ) from exc

                doc = fitz.open(str(path))
                try:
                    pages = [page.get_text("text") for page in doc]
                finally:
                    doc.close()
                return "\n\n".join(pages)

        reader = PdfReader(str(path))
        page_texts: list[str] = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            if extracted.strip():
                page_texts.append(extracted)
        return "\n\n".join(page_texts)