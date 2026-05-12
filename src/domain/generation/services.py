from __future__ import annotations

from pathlib import Path

from src.domain.generation.models import QuizGenerationDomainResult
from src.generator.orchestrator import GenerationOrchestrator


class QuizGenerationDomainService:
    def __init__(self, *, generator: GenerationOrchestrator | None = None) -> None:
        self._generator = generator or GenerationOrchestrator()

    def generate(
        self,
        *,
        plan_path: Path,
        output_dir: Path,
        artifact_root: Path,
        run_id: str,
        image_paths: list[str] | None,
        mock_image: bool,
        mock_question: bool,
    ) -> QuizGenerationDomainResult:
        payload = self._generator.run(
            plan_path,
            output_dir=output_dir,
            artifact_root=artifact_root,
            run_id=run_id,
            image_paths=image_paths,
            mock_image=mock_image,
            mock_question=mock_question,
        )
        return QuizGenerationDomainResult(payload=payload)
