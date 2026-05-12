from __future__ import annotations

from pathlib import Path

from src.application.contracts.workflow import QuizWorkflowRequest
from src.application.orchestrators.quiz_workflow_orchestrator import QuizWorkflowOrchestrator


class NotebookQuizApplicationService:
    """Application-layer entry point for notebook-triggered quiz workflows."""

    def __init__(
        self,
        *,
        orchestrator: QuizWorkflowOrchestrator,
        output_root: Path | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._output_root = output_root

    def generate_for_source(
        self,
        *,
        document_path: Path,
        num_questions: int,
        mock_image: bool,
        mock_question: bool,
        question_format_profile: dict[str, float] | None,
    ) -> dict[str, object]:
        result = self._orchestrator.run(
            QuizWorkflowRequest(
                document_path=document_path,
                output_root=self._output_root,
                num_questions=num_questions,
                mock_image=mock_image,
                mock_question=mock_question,
                question_format_profile=question_format_profile,
            )
        )
        return {
            "run_id": result.run_id,
            "run_root": result.run_root,
            "manifest": result.manifest,
            "log_path": result.log_path,
            "artifacts": result.artifacts,
            "stages": result.stages,
            "document": result.document,
            "extracted": result.extracted,
            "graph": result.graph,
            "graph_summary": result.graph_summary,
            "plans": result.plans,
            "generation": result.generation,
        }
