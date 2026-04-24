from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.knowledge.schema import MultimodalDocumentGraph, NodeKind
from src.planner.prompt_templates import render_planner_prompt
from src.utils.llm import LLMClient

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_OUTPUT = PROJECT_ROOT / "data" / "plans" / "quiz_plan.json"

@dataclass
class QuestionPlan:
    target_concept: str
    question_type: str
    difficulty: str
    reasoning_type: str
    image_role: str | None = None
    image_description: str | None = None
    learning_objective: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class QuizPlanner:
    """Create a plan before generating quiz content."""

    _VALID_DIFFICULTIES = {"easy", "medium", "hard"}
    _VALID_REASONING = {"factoid", "causal", "multi-hop"}
    _VALID_IMAGE_ROLES = {"illustrative", "reasoning", "distractor"}

    def __init__(
        self,
        knowledge_graph: MultimodalDocumentGraph | None = None,
        *,
        graph_json_path: str | Path | None = None,
        llm_client: LLMClient | Any | None = None,
        max_retries: int = 1,
    ) -> None:
        self._knowledge_graph = self._load_graph(knowledge_graph, graph_json_path)
        self._llm = llm_client or LLMClient()
        self._max_retries = max_retries

    def plan(
        self,
        *,
        num_questions: int,
        difficulty_distribution: dict[str, float],
    ) -> list[QuestionPlan]:
        if num_questions <= 0:
            raise ValueError("num_questions must be greater than zero.")

        self._validate_difficulty_distribution(difficulty_distribution)
        prompt = render_planner_prompt(
            graph_context=self._build_graph_context(),
            num_questions=num_questions,
            difficulty_distribution=difficulty_distribution,
        )

        last_error: Exception | None = None
        attempts = self._max_retries + 1
        def _clean_llm_output(raw: object) -> str:
            if raw is None:
                return ""
            s = str(raw).strip()

            # Remove fenced code blocks (```json ... ``` or ``` ... ```)
            if s.startswith("```"):
                # skip the opening fence
                # find the closing fence
                closing = s.rfind("```")
                if closing > 3:
                    inner = s[3:closing].strip()
                    # if inner starts with a language tag like 'json', strip it
                    if inner.lower().startswith("json"):
                        inner = inner[4:].lstrip()
                    s = inner

            # If still contains backticks, remove them
            s = s.strip().strip('`').strip()

            # Extract the first top-level JSON object by locating outermost braces
            first = s.find("{")
            last = s.rfind("}")
            if first != -1 and last != -1 and last > first:
                s = s[first:last+1]

            return s

        for _ in range(attempts):
            try:
                raw_output = self._llm.complete(prompt)
                # Quick sanity checks with clearer errors for debugging
                if raw_output is None or not str(raw_output).strip():
                    raise RuntimeError("LLM returned empty response")

                cleaned = _clean_llm_output(raw_output)
                if not cleaned:
                    print("[QuizPlanner] LLM returned only non-JSON content. Raw output repr:", repr(raw_output))
                    raise RuntimeError("LLM returned no JSON content")

                try:
                    payload = json.loads(cleaned)
                except json.JSONDecodeError:
                    print("[QuizPlanner] Failed to parse cleaned LLM output. Cleaned repr:", repr(cleaned))
                    print("[QuizPlanner] Original raw output repr:", repr(raw_output))
                    raise
                plans = self._parse_plans(payload, num_questions)
                self._repair_concept_coverage(plans)
                self._validate_concept_coverage(plans, num_questions)
                self._validate_difficulty_balance(plans, num_questions, difficulty_distribution)
                return plans
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            except RuntimeError as exc:
                last_error = exc
                break

        raise RuntimeError("Failed to generate a valid quiz plan.") from last_error

    def _build_graph_context(self) -> dict[str, object]:
        concept_nodes = [node for node in self._knowledge_graph.nodes if node.kind == NodeKind.concept]
        image_nodes = [node for node in self._knowledge_graph.nodes if node.kind == NodeKind.image]

        concepts = [
            {
                "id": node.id,
                "label": node.label,
            }
            for node in concept_nodes
        ]
        images = [
            {
                "id": node.id,
                "label": node.label,
                "image_path": node.image_path,
            }
            for node in image_nodes
        ]

        edges = [
            {
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation.value,
            }
            for edge in self._knowledge_graph.edges
        ]

        return {
            "summary": self._knowledge_graph.summary(),
            "concepts": concepts,
            "images": images,
            "edges": edges,
        }

    def _parse_plans(self, payload: Any, expected_count: int) -> list[QuestionPlan]:
        if not isinstance(payload, dict):
            raise RuntimeError("Planner output must be a JSON object.")
        rows = payload.get("questions")
        if not isinstance(rows, list):
            raise RuntimeError("Planner output must include a 'questions' list.")
        if len(rows) != expected_count:
            raise RuntimeError("Planner output question count does not match request.")

        plans: list[QuestionPlan] = []
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError("Each planned question must be an object.")

            target_concept = self._normalized_text(row.get("target_concept"))
            question_type = self._normalized_text(row.get("question_type"))
            difficulty = self._normalized_text(row.get("difficulty")).lower()
            reasoning_type = self._normalized_text(row.get("reasoning_type")).lower()
            # Images are mandatory for all generated questions
            # Provide sensible defaults if the planner did not include image fields
            raw_role = row.get("image_role")
            if raw_role is None:
                image_role_raw = "illustrative"
            else:
                normalized_role = str(raw_role).strip().lower()
                # Treat explicit 'none'/'null' or empty as missing and fall back
                if normalized_role in {"", "none", "null"}:
                    image_role_raw = "illustrative"
                else:
                    image_role_raw = raw_role

            image_description_raw = row.get("image_description") or target_concept
            image_role = self._normalize_image_role(image_role_raw)
            image_description = self._normalize_image_description(image_description_raw)
            learning_objective = self._normalized_text(row.get("learning_objective"))

            if difficulty not in self._VALID_DIFFICULTIES:
                raise RuntimeError(f"Unsupported difficulty: {difficulty}")
            if reasoning_type not in self._VALID_REASONING:
                raise RuntimeError(f"Unsupported reasoning_type: {reasoning_type}")

            plans.append(
                QuestionPlan(
                    target_concept=target_concept,
                    question_type=question_type,
                    difficulty=difficulty,
                    reasoning_type=reasoning_type,
                    image_role=image_role,
                    image_description=image_description,
                    learning_objective=learning_objective,
                )
            )

        return plans

    def _normalize_image_role(self, value: object) -> str:
        normalized = self._normalized_text(value).lower()
        if normalized not in self._VALID_IMAGE_ROLES:
            raise RuntimeError(f"Unsupported image_role: {normalized}")
        return normalized

    def _normalize_image_description(self, value: object) -> str:
        description = self._normalized_text(value)
        if not description:
            raise RuntimeError("image_description is required for all questions.")
        return description

    @staticmethod
    def _normalized_text(value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _validate_concept_coverage(self, plans: list[QuestionPlan], num_questions: int) -> None:
        concepts = [plan.target_concept.casefold() for plan in plans]
        if len(set(concepts)) != num_questions:
            raise RuntimeError("Concept coverage violated: duplicate target concepts found.")

    def _repair_concept_coverage(self, plans: list[QuestionPlan]) -> None:
        """Best-effort repair for duplicate/empty target concepts.

        Strategy:
        1) Replace duplicates with unused concept labels from the KG.
        2) If KG candidates are exhausted, suffix the original label to keep IDs unique.
        """
        seen: set[str] = set()
        duplicate_indexes: list[int] = []

        for idx, plan in enumerate(plans):
            key = self._normalized_text(plan.target_concept).casefold()
            if not key or key in seen:
                duplicate_indexes.append(idx)
            else:
                seen.add(key)

        if not duplicate_indexes:
            return

        candidate_labels: list[str] = []
        candidate_seen: set[str] = set()
        for node in self._knowledge_graph.nodes:
            if node.kind != NodeKind.concept:
                continue
            label = self._normalized_text(node.label)
            key = label.casefold()
            if not label or key in seen or key in candidate_seen:
                continue
            candidate_labels.append(label)
            candidate_seen.add(key)

        for idx in duplicate_indexes:
            plan = plans[idx]
            previous = self._normalized_text(plan.target_concept)

            if candidate_labels:
                replacement = candidate_labels.pop(0)
            else:
                replacement = self._make_unique_label(previous or "concept", seen)

            plan.target_concept = replacement
            prev_desc = self._normalized_text(plan.image_description)
            if not prev_desc or prev_desc.casefold() == previous.casefold():
                plan.image_description = replacement

            seen.add(replacement.casefold())

    def _make_unique_label(self, base: str, seen: set[str]) -> str:
        root = self._normalized_text(base) or "concept"
        if root.casefold() not in seen:
            return root

        suffix = 2
        while True:
            candidate = f"{root}_{suffix}"
            if candidate.casefold() not in seen:
                return candidate
            suffix += 1

    def _validate_difficulty_distribution(self, distribution: dict[str, float]) -> None:
        if not distribution:
            raise ValueError("difficulty_distribution cannot be empty.")
        unknown = {key for key in distribution if key not in self._VALID_DIFFICULTIES}
        if unknown:
            raise ValueError(f"Unknown difficulty buckets: {sorted(unknown)}")

    def _validate_difficulty_balance(
        self,
        plans: list[QuestionPlan],
        num_questions: int,
        distribution: dict[str, float],
    ) -> None:
        expected_counts = self._expected_difficulty_counts(num_questions, distribution)
        actual_counts = {"easy": 0, "medium": 0, "hard": 0}
        for plan in plans:
            actual_counts[plan.difficulty] += 1
        # Allow a small tolerance per bucket to account for LLM variability
        tolerance = 2
        violations = {
            k: (expected_counts[k], actual_counts.get(k, 0))
            for k in expected_counts
            if abs(actual_counts.get(k, 0) - expected_counts[k]) > tolerance
        }

        if violations:
            raise RuntimeError(
                "Difficulty balancing violated within tolerance: "
                f"expected {expected_counts}, got {actual_counts}, tolerance={tolerance}. Violations: {violations}"
            )

    def _expected_difficulty_counts(
        self,
        num_questions: int,
        distribution: dict[str, float],
    ) -> dict[str, int]:
        buckets = ["easy", "medium", "hard"]
        scaled = {bucket: max(distribution.get(bucket, 0.0), 0.0) * num_questions for bucket in buckets}
        counts = {bucket: int(scaled[bucket]) for bucket in buckets}
        remainder = num_questions - sum(counts.values())
        fractions = sorted(
            buckets,
            key=lambda bucket: (scaled[bucket] - counts[bucket]),
            reverse=True,
        )
        for bucket in fractions[:remainder]:
            counts[bucket] += 1
        return counts
    
    def _load_graph(
        self,
        graph: Optional[MultimodalDocumentGraph] = None,
        json_path: Optional[str | Path] = None,
    ) -> MultimodalDocumentGraph:
        """
        Returns a MultimodalDocumentGraph from memory or JSON fallback.
        """

        # 1. If already in memory → use it
        if graph is not None:
            return graph

        # 2. Otherwise load from JSON
        if json_path is None:
            raise ValueError("Either graph or json_path must be provided.")

        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(f"Graph JSON not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return MultimodalDocumentGraph(**data)
    
    def save_plan(
        self,
        plans: list["QuestionPlan"],
        output_path: str | Path = DEFAULT_PLAN_OUTPUT,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [asdict(p) for p in plans]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[QuizPlanner] Saved JSON plan → {output_path}")

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Test QuizPlanner")

    # -----------------------------
    # INPUT GRAPH
    # -----------------------------
    parser.add_argument(
        "--graph_path",
        type=str,
        required=True,
        help="Path to KG JSON file"
    )

    parser.add_argument(
        "--use_memory",
        action="store_true",
        help="Use in-memory dummy graph instead of JSON"
    )

    # -----------------------------
    # QUIZ CONFIG
    # -----------------------------
    parser.add_argument(
        "--num_questions",
        type=int,
        default=5,
        help="Number of quiz questions"
    )

    parser.add_argument(
        "--easy",
        type=float,
        default=0.4,
        help="Easy difficulty ratio"
    )

    parser.add_argument(
        "--medium",
        type=float,
        default=0.4,
        help="Medium difficulty ratio"
    )

    parser.add_argument(
        "--hard",
        type=float,
        default=0.2,
        help="Hard difficulty ratio"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PLAN_OUTPUT,
        help="Path to write the generated plan JSON",
    )

    args = parser.parse_args()

    # -----------------------------
    # BUILD DIFFICULTY DISTRIBUTION
    # -----------------------------
    difficulty_distribution = {
        "easy": args.easy,
        "medium": args.medium,
        "hard": args.hard
    }

    # -----------------------------
    # LOAD GRAPH
    # -----------------------------
    graph = None

    if args.use_memory:
        print("Using in-memory graph...")

        graph = MultimodalDocumentGraph(
            document_id="test",
            nodes=[],
            edges=[]
        )

    else:
        graph_path = Path(args.graph_path)

        print(f"Loading graph from: {graph_path}")

        if not graph_path.exists():
            raise FileNotFoundError(f"Graph not found: {graph_path}")

        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        graph = MultimodalDocumentGraph(**data)

    # -----------------------------
    # INIT PLANNER
    # -----------------------------
    print("Initializing planner...")

    planner = QuizPlanner(
        knowledge_graph=graph,
        max_retries=2
    )

    # -----------------------------
    # RUN PLANNING
    # -----------------------------
    print("Generating quiz plan...\n")

    plans = planner.plan(
        num_questions=args.num_questions,
        difficulty_distribution=difficulty_distribution
    )

    planner.save_plan(plans, args.output)
    return 0


def load_plan(plan_path: str | Path) -> list[QuestionPlan]:
    """Load a saved plan JSON (list of question objects) and return QuestionPlan instances.

    This is a lightweight loader used by scripts that consume planner output.
    It fills sensible defaults for missing `image_role`/`image_description`.
    """
    path = Path(plan_path)
    if not path.exists():
        raise FileNotFoundError(f"Plan JSON not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise RuntimeError("Plan JSON must be a list of question objects.")

    plans: list[QuestionPlan] = []
    for item in data:
        if not isinstance(item, dict):
            raise RuntimeError("Each plan item must be an object.")

        target_concept = str(item.get("target_concept", "")).strip()
        question_type = str(item.get("question_type", "multiple_choice")).strip()
        difficulty = str(item.get("difficulty", "medium")).strip()
        reasoning_type = str(item.get("reasoning_type", "factoid")).strip()

        image_role = item.get("image_role") or "illustrative"
        image_description = item.get("image_description") or target_concept
        learning_objective = item.get("learning_objective")
        metadata = item.get("metadata", {}) or {}

        plans.append(
            QuestionPlan(
                target_concept=target_concept,
                question_type=question_type,
                difficulty=difficulty,
                reasoning_type=reasoning_type,
                image_role=image_role,
                image_description=image_description,
                learning_objective=learning_objective,
                metadata=metadata,
            )
        )

    return plans

if __name__ == "__main__":
    raise SystemExit(main())
