from __future__ import annotations

from pathlib import Path

from src.application.contracts.workflow import QuizGenerationResult
from src.application.services.artifact_service import RunArtifactService
from src.application.services.run_context import RunContext
from src.domain.generation.services import QuizGenerationDomainService


class QuizGenerationService:
    def __init__(
        self,
        *,
        artifact_service: RunArtifactService,
        domain_service: QuizGenerationDomainService | None = None,
    ) -> None:
        self._artifact_service = artifact_service
        self._domain_service = domain_service or QuizGenerationDomainService()

    def generate(
        self,
        *,
        context: RunContext,
        image_paths: list[str] | None,
        mock_image: bool,
        mock_question: bool,
    ) -> QuizGenerationResult:
        plan_path = context.planning_dir / "quiz_plan.json"
        generation_result = self._domain_service.generate(
            plan_path=plan_path,
            output_dir=context.generation_dir,
            artifact_root=context.run_root,
            run_id=context.run_id,
            image_paths=image_paths,
            mock_image=mock_image,
            mock_question=mock_question,
        )
        payload = generation_result.payload

        artifacts: dict[str, str] = {}
        generation_artifacts = payload.get("artifacts", {})
        if isinstance(generation_artifacts, dict):
            for key, value in generation_artifacts.items():
                if isinstance(value, Path):
                    artifacts[f"generation_{key}"] = self._artifact_service.project_relative(value)

        questions_path = context.generation_dir / "questions.json"
        quiz_package_path = context.generation_dir / "quiz_package.json"
        image_artifacts_path = context.generation_dir / "image_artifacts.json"
        artifacts["questions"] = self._artifact_service.project_relative(questions_path)
        artifacts["quiz_package"] = self._artifact_service.project_relative(quiz_package_path)
        artifacts["image_artifacts"] = self._artifact_service.project_relative(image_artifacts_path)

        return QuizGenerationResult(payload=payload, artifacts=artifacts)
