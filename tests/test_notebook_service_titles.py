"""Unit tests for quiz title generation in NotebookService."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from api.services import NotebookService
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-specific dependency gaps
    NotebookService = None
    IMPORT_ERROR = exc


@unittest.skipIf(NotebookService is None, f"NotebookService dependencies unavailable: {IMPORT_ERROR}")
class TestNotebookServiceQuizTitles(unittest.TestCase):
    def _service_with_llm(self, llm_response: str | Exception) -> NotebookService:
        service = NotebookService.__new__(NotebookService)
        if isinstance(llm_response, Exception):
            client = Mock()
            client.complete.side_effect = llm_response
            service._llm_client = client
            return service
        client = Mock()
        client.complete.return_value = llm_response
        service._llm_client = client
        return service

    def _sample_quiz_package(self) -> dict[str, object]:
        return {
            "results": [
                {
                    "question": {
                        "target_concept": "Photosynthesis",
                        "question_text": "Which stage uses light-dependent reactions?",
                    }
                },
                {
                    "question": {
                        "target_concept": "Chloroplast",
                        "question_text": "What is the role of thylakoid membranes?",
                    }
                },
            ]
        }

    def test_generate_quiz_title_uses_llm_response(self) -> None:
        service = self._service_with_llm(' "Photosynthesis Mastery Challenge!" ')
        title = service._generate_quiz_title(
            self._sample_quiz_package(),
            source_title="Plant Biology Chapter 3",
            notebook_title="Botany 101",
        )
        self.assertEqual(title, "Photosynthesis Mastery Challenge")

    def test_generate_quiz_title_falls_back_on_llm_failure(self) -> None:
        service = self._service_with_llm(RuntimeError("network down"))
        title = service._generate_quiz_title(
            self._sample_quiz_package(),
            source_title="Plant Biology Chapter 3",
            notebook_title="Botany 101",
        )
        self.assertEqual(title, "Plant Biology Chapter 3 Quiz")

    def test_generate_quiz_title_uses_notebook_title_fallback(self) -> None:
        service = self._service_with_llm("")
        title = service._generate_quiz_title(
            self._sample_quiz_package(),
            source_title="",
            notebook_title="Botany 101",
        )
        self.assertEqual(title, "Botany 101 Quiz")

    def test_sanitize_generated_title_enforces_word_and_length_caps(self) -> None:
        service = self._service_with_llm("unused")
        long_title = (
            "This title has way too many words and should be trimmed "
            "to six words only for consistency!"
        )
        sanitized = service._sanitize_generated_title(long_title)
        self.assertEqual(sanitized, "This title has way too many")
        self.assertLessEqual(len(sanitized), 120)


if __name__ == "__main__":
    unittest.main()
