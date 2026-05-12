"""Unit tests for question format profiles and Question validation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.knowledge.schema import Question
from src.question_formats import (
    QuestionFormatProfile,
    coerce_format_profile,
    merge_question_metadata,
    normalize_question_type,
    parse_question_formats_json,
)


class TestNormalize(unittest.TestCase):
    def test_aliases(self) -> None:
        self.assertEqual(normalize_question_type("MCQ"), "multiple_choice")
        self.assertEqual(normalize_question_type("true-false"), "true_false")
        self.assertEqual(normalize_question_type("fill in blank"), "fill_in_blank")


class TestProfile(unittest.TestCase):
    def test_coerce_default(self) -> None:
        p = coerce_format_profile(None)
        self.assertEqual(dict(p.distribution), {"multiple_choice": 1.0})

    def test_coerce_string(self) -> None:
        p = coerce_format_profile("true_false")
        self.assertEqual(dict(p.distribution), {"true_false": 1.0})

    def test_json_parse(self) -> None:
        p = parse_question_formats_json('{"multiple_choice": 0.5, "true_false": 0.5}')
        self.assertEqual(len(p.allowed_types), 2)
        counts = p.expected_counts(10)
        self.assertEqual(sum(counts.values()), 10)

    def test_distribution_tolerance(self) -> None:
        p = QuestionFormatProfile(distribution={"multiple_choice": 0.5, "true_false": 0.5})
        self.assertTrue(p.distribution_within_tolerance({"multiple_choice": 5, "true_false": 5}, 10))
        self.assertFalse(p.distribution_within_tolerance({"multiple_choice": 10, "true_false": 0}, 10))


class TestQuestionValidate(unittest.TestCase):
    def _base(self, **kwargs: object) -> Question:
        data = {
            "id": "q1",
            "question_text": "Sample?",
            "options": [],
            "correct_answer": "x",
            "explanation": "Because.",
            "target_concept": "c",
            "difficulty": "easy",
            "question_type": "multiple_choice",
            "associated_image": "/img.png",
        }
        data.update(kwargs)
        return Question(**data)

    def test_mcq_valid(self) -> None:
        q = self._base(
            question_type="multiple_choice",
            options=["a", "b", "c", "d"],
            correct_answer="a",
            question_text="Pick one.",
        )
        q.validate()

    def test_true_false(self) -> None:
        q = self._base(
            question_type="true_false",
            options=["True", "False"],
            correct_answer="False",
            question_text="The sky is green.",
        )
        q.validate()

    def test_fill_in_blank(self) -> None:
        q = self._base(
            question_type="fill_in_blank",
            correct_answer="photosynthesis",
            question_text="Plants use ____ to make food.",
            options=[],
        )
        q.validate()

    def test_matching(self) -> None:
        q = self._base(
            question_type="matching",
            question_text="Match each term to its definition.",
            correct_answer="A→1; B→2",
            options=[],
            metadata={
                "matching_left": ["A", "B"],
                "matching_right": ["1", "2"],
                "matching_solution": [[0, 1], [1, 0]],
            },
        )
        q.validate()

    def test_matching_invalid(self) -> None:
        q = self._base(
            question_type="matching",
            question_text="Match.",
            correct_answer="bad",
            options=[],
            metadata={
                "matching_left": ["A", "B"],
                "matching_right": ["1", "2"],
                "matching_solution": [[0, 0], [1, 0]],
            },
        )
        with self.assertRaises(ValueError):
            q.validate()


class TestMergeMetadata(unittest.TestCase):
    def test_matching_keys(self) -> None:
        base = {"a": 1}
        payload = {"matching_left": ["x"], "matching_right": ["y"], "matching_solution": [[0, 0]]}
        out = merge_question_metadata(base, payload, question_type="matching")
        self.assertIn("matching_left", out)
        self.assertEqual(out["a"], 1)


if __name__ == "__main__":
    unittest.main()
