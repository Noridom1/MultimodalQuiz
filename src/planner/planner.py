from __future__ import annotations

from dataclasses import dataclass, field

from src.knowledge.schema import KnowledgeGraph


@dataclass
class QuestionPlan:
    target_concept: str
    question_type: str
    difficulty: str
    reasoning_type: str
    requires_image: bool
    image_role: str | None = None
    image_description: str | None = None
    learning_objective: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class QuizPlanner:
    """Create a plan before generating quiz content."""

    def plan(
        self,
        knowledge_graph: KnowledgeGraph,
        *,
        num_questions: int,
        difficulty_distribution: dict[str, float],
    ) -> list[QuestionPlan]:
        raise NotImplementedError("Quiz planning is not implemented yet.")
