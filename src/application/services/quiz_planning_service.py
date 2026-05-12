from __future__ import annotations

from src.application.contracts.workflow import KnowledgeGraphResult, QuizPlanResult
from src.application.services.artifact_service import RunArtifactService
from src.application.services.run_context import RunContext
from src.domain.knowledge_graph.models import KnowledgeGraphDomainResult
from src.domain.planning.services import QuizPlanningDomainService
from src.question_formats import QuestionFormatProfile


class QuizPlanningService:
    def __init__(
        self,
        *,
        artifact_service: RunArtifactService,
        domain_service: QuizPlanningDomainService | None = None,
    ) -> None:
        self._artifact_service = artifact_service
        self._domain_service = domain_service or QuizPlanningDomainService()

    def plan(
        self,
        *,
        context: RunContext,
        graph_result: KnowledgeGraphResult,
        generation_mode: str,
        num_questions: int,
        difficulty_distribution: dict[str, float],
        format_profile: QuestionFormatProfile,
    ) -> QuizPlanResult:
        domain_result = self._domain_service.plan(
            graph_result=KnowledgeGraphDomainResult(graph_result=graph_result.graph_result),
            generation_mode=generation_mode,
            num_questions=num_questions,
            difficulty_distribution=difficulty_distribution,
            format_profile=format_profile,
        )
        plans = domain_result.plans
        plan_path = context.planning_dir / "quiz_plan.json"
        self._domain_service.save_plan(plans, plan_path)
        return QuizPlanResult(
            plans=plans,
            planner_name=domain_result.planner_name,
            artifacts={"plan": self._artifact_service.project_relative(plan_path)},
        )
