from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any

from src.utils.llm import LLMClient

_CONCEPT_LIST_JSON_PATTERN = re.compile(r"\[[\s\S]*\]")

DOCUMENT_CONCEPT_PROMPT = """Given the following passage, extract a comprehensive list of domain concepts.\nReturn a JSON list of short concept phrases (3 words or fewer each).\nCover both explicit and implied concepts, but avoid filler terms.\n\nPassage:\n{content}"""

QUESTION_CONCEPT_PROMPT = """Given the following quiz question, extract the key domain concepts being tested.\nReturn a JSON list of short concept phrases (3 words or fewer each).\nDo not include quiz mechanics such as \"which of the following\".\n\nQuestion:\n{question}\n\nChoices:\n{choices}"""

SYSTEM_PROMPT = (
    "You are an educational concept extraction assistant. "
    "Always return strict JSON matching the requested schema, with no markdown fences."
)


def _is_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retry_signals = (
        "rate limit",
        "ratelimit",
        "too many requests",
        "timeout",
        "timed out",
        "temporar",
        "unavailable",
        "503",
        "429",
        "connection",
        "reset",
    )
    return any(signal in message for signal in retry_signals)


def _complete_with_retry(
    llm_client: LLMClient,
    prompt: str,
    *,
    system_prompt: str,
    max_retries: int,
    initial_backoff_seconds: float,
    max_backoff_seconds: float,
) -> str:
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return llm_client.complete(prompt, system_prompt=system_prompt)
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries or not _is_retryable_error(exc):
                raise

            sleep_seconds = min(max_backoff_seconds, initial_backoff_seconds * (2**attempt))
            jitter = random.uniform(0.0, min(1.0, sleep_seconds * 0.1))
            time.sleep(sleep_seconds + jitter)

    raise RuntimeError("LLM completion failed after retries") from last_error


def load_document_text(document_path: str | Path) -> str:
    """Load a raw document into plain text for concept extraction."""
    path = Path(document_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if path.suffix.lower() == ".pdf":
        text = _read_pdf_text(path)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    text = text.strip()
    if not text:
        raise ValueError(f"Document is empty: {path}")

    return text


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


def _extract_json_list(raw: str) -> list[str]:
    if raw is None:
        return []

    text = str(raw).strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
    except json.JSONDecodeError:
        pass

    match = _CONCEPT_LIST_JSON_PATTERN.search(text)
    if not match:
        return []

    try:
        payload = json.loads(match.group(0))
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
    except json.JSONDecodeError:
        return []

    return []


def _normalize_concepts(concepts: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()

    for concept in concepts:
        normalized = re.sub(r"\s+", " ", concept).strip(" \t\n\r.,;:!?\"'").lower()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)

    return cleaned


def _chunk_text(text: str, max_chars: int = 6000) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    cursor = 0
    while cursor < len(text):
        chunk = text[cursor : cursor + max_chars]
        chunks.append(chunk)
        cursor += max_chars
    return chunks


def extract_document_concepts(
    document_text: str,
    llm_client: LLMClient,
    *,
    chunk_chars: int = 6000,
    max_retries: int = 5,
    initial_backoff_seconds: float = 1.0,
    max_backoff_seconds: float = 20.0,
) -> list[str]:
    """Extract concept phrases from a document using the configured LLM."""
    concepts: list[str] = []
    for chunk in _chunk_text(document_text, max_chars=chunk_chars):
        prompt = DOCUMENT_CONCEPT_PROMPT.format(content=chunk)
        raw = _complete_with_retry(
            llm_client,
            prompt,
            system_prompt=SYSTEM_PROMPT,
            max_retries=max_retries,
            initial_backoff_seconds=initial_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
        )
        concepts.extend(_extract_json_list(raw))
    return _normalize_concepts(concepts)


def _coerce_question_text(question_item: dict[str, Any]) -> tuple[str, list[str]]:
    question_text = str(question_item.get("question_text") or question_item.get("question") or "").strip()
    choices_obj = question_item.get("options")
    if choices_obj is None:
        choices_obj = question_item.get("choices")

    if isinstance(choices_obj, list):
        choices = [str(item).strip() for item in choices_obj if str(item).strip()]
    else:
        choices = []

    return question_text, choices


def extract_question_concepts(
    question_items: list[dict[str, Any]],
    llm_client: LLMClient,
    *,
    max_retries: int = 5,
    initial_backoff_seconds: float = 1.0,
    max_backoff_seconds: float = 20.0,
) -> dict[str, Any]:
    """Extract concept phrases for each question and return a merged set."""
    per_question: list[dict[str, Any]] = []
    merged: list[str] = []

    for idx, item in enumerate(question_items, start=1):
        question_text, choices = _coerce_question_text(item)
        if not question_text:
            per_question.append({"index": idx, "question": "", "concepts": []})
            continue

        prompt = QUESTION_CONCEPT_PROMPT.format(
            question=question_text,
            choices="; ".join(choices) if choices else "(none)",
        )
        raw = _complete_with_retry(
            llm_client,
            prompt,
            system_prompt=SYSTEM_PROMPT,
            max_retries=max_retries,
            initial_backoff_seconds=initial_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
        )
        concepts = _normalize_concepts(_extract_json_list(raw))
        merged.extend(concepts)
        per_question.append(
            {
                "index": idx,
                "question": question_text,
                "concepts": concepts,
            }
        )

    return {
        "per_question": per_question,
        "all_concepts": _normalize_concepts(merged),
    }
