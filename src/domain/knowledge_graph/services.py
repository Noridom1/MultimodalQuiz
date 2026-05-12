from __future__ import annotations

from pathlib import Path

from src.domain.document_processing.models import DocumentProcessingDomainResult
from src.domain.knowledge_graph.models import KnowledgeGraphDomainResult
from src.knowledge.kg_builder import build_knowledge_graph_workflow


class KnowledgeGraphDomainService:
    def build(
        self,
        *,
        document_path: Path,
        processed: DocumentProcessingDomainResult,
        chunk_max_tokens: int,
        overlap_blocks: int,
    ) -> KnowledgeGraphDomainResult:
        parsed_document = processed.parsed_document
        graph_result = build_knowledge_graph_workflow(
            {
                "markdown": parsed_document.markdown,
                "sections": parsed_document.sections,
                "paragraphs": parsed_document.paragraphs,
                "figures": parsed_document.figures,
                "captions": parsed_document.captions,
            },
            processed.extracted,
            source_file=document_path,
            max_tokens=chunk_max_tokens,
            overlap_blocks=overlap_blocks,
        )
        return KnowledgeGraphDomainResult(graph_result=graph_result)
