from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.planner.planner import QuestionPlan
from src.planner.topic_planner import TopicAgenticPlanner
from src.question_formats import QuestionFormatProfile


class TestTopicPlannerTypeDistribution(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = object.__new__(TopicAgenticPlanner)
        self.profile = QuestionFormatProfile(
            distribution={
                "multiple_choice": 0.25,
                "true_false": 0.25,
                "fill_in_blank": 0.25,
                "matching": 0.25,
            }
        )

    def test_required_types_for_low_count_batches(self) -> None:
        target = self.profile.expected_counts(5)
        produced = Counter()

        first = self.planner._build_required_types_for_batch(
            profile=self.profile, target_counts=target, produced_counts=produced, batch_size=1
        )
        produced.update(first)
        second = self.planner._build_required_types_for_batch(
            profile=self.profile, target_counts=target, produced_counts=produced, batch_size=1
        )
        produced.update(second)
        third = self.planner._build_required_types_for_batch(
            profile=self.profile, target_counts=target, produced_counts=produced, batch_size=1
        )
        produced.update(third)
        fourth = self.planner._build_required_types_for_batch(
            profile=self.profile, target_counts=target, produced_counts=produced, batch_size=1
        )
        produced.update(fourth)
        fifth = self.planner._build_required_types_for_batch(
            profile=self.profile, target_counts=target, produced_counts=produced, batch_size=1
        )
        produced.update(fifth)

        self.assertEqual(sum(produced.values()), 5)
        self.assertEqual(dict(produced), target)

    def test_required_types_for_ten_questions(self) -> None:
        target = self.profile.expected_counts(10)
        produced = Counter()
        for _ in range(10):
            batch = self.planner._build_required_types_for_batch(
                profile=self.profile, target_counts=target, produced_counts=produced, batch_size=1
            )
            produced.update(batch)

        self.assertEqual(sum(produced.values()), 10)
        self.assertEqual(dict(produced), target)

    def test_final_hard_check_repairs_true_false_collapse(self) -> None:
        collapsed = [
            QuestionPlan(
                target_concept=f"c{i}",
                question_type="true_false",
                difficulty="easy",
                reasoning_type="factoid",
                image_role="illustrative",
                image_description="img",
                learning_objective="obj",
                tested_fact_block_id=f"b{i}",
                metadata={},
            )
            for i in range(5)
        ]
        repaired = self.planner._finalize_global_type_distribution(
            plans=collapsed, profile=self.profile, requested_total=5
        )
        counts = Counter(p.question_type for p in repaired)
        self.assertEqual(dict(counts), self.profile.expected_counts(5))
        self.assertGreater(len(counts), 1)

    def test_parse_enforces_required_type_sequence(self) -> None:
        payload = {
            "questions": [
                {
                    "target_concept": "A",
                    "question_type": "true_false",
                    "difficulty": "easy",
                    "reasoning_type": "factoid",
                    "image_role": "illustrative",
                    "image_description": "d",
                    "learning_objective": "o",
                    "tested_fact_block_id": "b1",
                    "metadata": {},
                }
            ]
        }
        with self.assertRaises(RuntimeError):
            self.planner._parse_topic_plans(
                payload=payload,
                expected_count=1,
                topic_id="t1",
                format_profile=self.profile,
                required_types=["multiple_choice"],
            )


if __name__ == "__main__":
    unittest.main()
