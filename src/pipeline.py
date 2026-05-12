from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from src.application.contracts.workflow import QuizWorkflowRequest
from src.application.orchestrators.quiz_workflow_orchestrator import (
    DEFAULT_DIFFICULTY_DISTRIBUTION,
    PROJECT_ROOT,
    QuizWorkflowOrchestrator,
    resolve_question_format_profile,
)
from src.question_formats import QuestionFormatProfile


@dataclass
class QuizGenerationPipeline:
    """Compatibility facade over the application-layer workflow orchestrator."""

    output_root: str | Path | None = None
    html_graph: bool = True

    def run(
        self,
        document_path: str | Path,
        *,
        output_root: str | Path | None = None,
        run_id: str | None = None,
        num_questions: int = 5,
        difficulty_distribution: dict[str, float] | None = None,
        image_paths: list[str] | None = None,
        mock_image: bool = False,
        mock_question: bool = False,
        generation_mode: str = "topic_agentic",
        question_format_profile: QuestionFormatProfile | dict[str, float] | str | None = None,
    ) -> dict[str, object]:
        orchestrator = QuizWorkflowOrchestrator(
            project_root=PROJECT_ROOT,
            html_graph=self.html_graph,
        )
        result = orchestrator.run(
            QuizWorkflowRequest(
                document_path=Path(document_path),
                output_root=Path(output_root) if output_root is not None else (
                    Path(self.output_root) if self.output_root is not None else None
                ),
                run_id=run_id,
                num_questions=num_questions,
                difficulty_distribution=difficulty_distribution,
                image_paths=image_paths,
                mock_image=mock_image,
                mock_question=mock_question,
                generation_mode=generation_mode,
                question_format_profile=question_format_profile,
                html_graph=self.html_graph,
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


__all__ = [
    "DEFAULT_DIFFICULTY_DISTRIBUTION",
    "PROJECT_ROOT",
    "QuizGenerationPipeline",
    "QuizWorkflowOrchestrator",
    "QuizWorkflowRequest",
    "resolve_question_format_profile",
    "asdict",
]
