from __future__ import annotations

from pathlib import Path

from src.application.contracts.workflow import DocumentProcessingResult
from src.application.services.artifact_service import RunArtifactService
from src.application.services.run_context import RunContext
from src.domain.document_processing.services import DocumentProcessingDomainService


class DocumentProcessingService:
    def __init__(
        self,
        *,
        artifact_service: RunArtifactService,
        domain_service: DocumentProcessingDomainService | None = None,
    ) -> None:
        self._artifact_service = artifact_service
        self._domain_service = domain_service or DocumentProcessingDomainService()

    def process(
        self,
        *,
        context: RunContext,
        document_path: Path,
        extractor_backend: str,
        extractor_provider: str,
        extractor_granularity: str,
        extractor_model: str | None,
        kg_chunk_max_tokens: int,
        kg_overlap_blocks: int,
    ) -> DocumentProcessingResult:
        domain_result = self._domain_service.process(
            document_path=document_path,
            document_dir=context.document_dir,
            extraction_dir=context.extraction_dir,
            extractor_backend=extractor_backend,
            extractor_provider=extractor_provider,
            extractor_granularity=extractor_granularity,
            extractor_model=extractor_model,
            chunk_max_tokens=kg_chunk_max_tokens,
            overlap_blocks=kg_overlap_blocks,
        )

        return DocumentProcessingResult(
            parsed_document=domain_result.parsed_document,
            extracted=domain_result.extracted,
            artifacts={
                "parsed_document": self._artifact_service.project_relative(domain_result.parsed_path),
                "extracted": self._artifact_service.project_relative(domain_result.extracted_path),
            },
        )
