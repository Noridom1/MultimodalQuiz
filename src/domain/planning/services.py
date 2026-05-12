from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.domain.knowledge_graph.models import KnowledgeGraphDomainResult
from src.domain.planning.models import QuizPlanDomainResult
from src.planner.planner import QuizPlanner
from src.planner.topic_planner import TopicAgenticPlanner
from src.question_formats import QuestionFormatProfile


class QuizPlanningDomainService:
    def plan(
        self,
        *,
        graph_result: KnowledgeGraphDomainResult,
        generation_mode: str,
        num_questions: int,
        difficulty_distribution: dict[str, float],
        format_profile: QuestionFormatProfile,
    ) -> QuizPlanDomainResult:
        document_graph = graph_result.graph_result.graph
        if generation_mode == "topic_agentic":
            planner = TopicAgenticPlanner(knowledge_graph=document_graph)
            planner_name = "topic_agentic"
        else:
            planner = QuizPlanner(knowledge_graph=document_graph)
            planner_name = "legacy"

        plans = planner.plan(
            num_questions=num_questions,
            difficulty_distribution=difficulty_distribution,
            format_profile=format_profile,
        )
        return QuizPlanDomainResult(plans=plans, planner_name=planner_name)

    def save_plan(self, plans: list[object], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(plan) for plan in plans], f, indent=2, ensure_ascii=False)
