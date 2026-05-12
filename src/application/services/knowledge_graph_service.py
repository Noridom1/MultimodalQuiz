from __future__ import annotations

from pathlib import Path

from src.application.contracts.workflow import DocumentProcessingResult, KnowledgeGraphResult
from src.application.services.artifact_service import RunArtifactService
from src.application.services.run_context import RunContext
from src.domain.document_processing.models import DocumentProcessingDomainResult
from src.domain.knowledge_graph.services import KnowledgeGraphDomainService
from src.knowledge.kg_builder import export_graph_bundle


class KnowledgeGraphService:
    def __init__(
        self,
        *,
        artifact_service: RunArtifactService,
        html_graph: bool = True,
        domain_service: KnowledgeGraphDomainService | None = None,
    ) -> None:
        self._artifact_service = artifact_service
        self._html_graph = html_graph
        self._domain_service = domain_service or KnowledgeGraphDomainService()

    def build(
        self,
        *,
        context: RunContext,
        document_path: Path,
        processed: DocumentProcessingResult,
        kg_chunk_max_tokens: int,
        kg_overlap_blocks: int,
    ) -> KnowledgeGraphResult:
        domain_result = self._domain_service.build(
            document_path=document_path,
            processed=DocumentProcessingDomainResult(
                parsed_document=processed.parsed_document,
                extracted=processed.extracted,
                parsed_path=context.document_dir / "parsed_document.json",
                extracted_path=context.extraction_dir / "extracted.json",
            ),
            chunk_max_tokens=kg_chunk_max_tokens,
            overlap_blocks=kg_overlap_blocks,
        )
        graph_result = domain_result.graph_result
        document_graph = graph_result.graph
        graph_export = export_graph_bundle(
            document_graph,
            output_dir=context.graph_dir,
            html=self._html_graph,
            checkpoints=graph_result.checkpoints,
        )
        graph_json_path = self._artifact_service.rename_artifact(
            graph_export["graph_json"],
            context.graph_dir / "graph.json",
        )
        networkx_json_path = self._artifact_service.rename_artifact(
            graph_export["networkx_json"],
            context.graph_dir / "graph_networkx.json",
        )
        html_path = None
        if "html" in graph_export:
            html_path = self._artifact_service.rename_artifact(
                graph_export["html"],
                context.graph_dir / "graph.html",
            )

        artifacts = {
            "graph": self._artifact_service.project_relative(graph_json_path),
            "graph_networkx": self._artifact_service.project_relative(networkx_json_path),
        }
        if html_path is not None:
            artifacts["graph_html"] = self._artifact_service.project_relative(html_path)

        for key in (
            "hierarchy",
            "chunks",
            "artifact_links",
            "extraction_raw",
            "canonicalization",
            "merge_review",
            "merge_application",
            "graph_consolidated",
            "topic_candidates",
            "topic_consolidation",
            "topics",
            "graph_validation",
        ):
            if key in graph_export:
                artifacts[key] = self._artifact_service.project_relative(graph_export[key])

        return KnowledgeGraphResult(graph_result=graph_result, artifacts=artifacts)
