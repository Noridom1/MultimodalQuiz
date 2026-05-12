from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from src.document_understanding.chunking import build_semantic_chunks, parse_markdown_blocks
from src.document_understanding.extractor import DocumentExtractor
from src.document_understanding.parser import parse_document
from src.domain.document_processing.models import DocumentProcessingDomainResult
from src.utils.io import write_json


def empty_extracted_payload() -> dict[str, object]:
    return {
        "concepts": [],
        "definitions": {},
        "relations": [],
        "examples": [],
    }


class DocumentProcessingDomainService:
    def process(
        self,
        *,
        document_path: Path,
        document_dir: Path,
        extraction_dir: Path,
        extractor_backend: str,
        extractor_provider: str,
        extractor_granularity: str,
        extractor_model: str | None,
        chunk_max_tokens: int,
        overlap_blocks: int,
    ) -> DocumentProcessingDomainResult:
        parsed_document = parse_document(document_path)
        parsed_path = document_dir / "parsed_document.json"
        write_json(parsed_path, asdict(parsed_document))

        chunk_blocks = parse_markdown_blocks(parsed_document.markdown, source_file=document_path)
        semantic_chunks = build_semantic_chunks(
            chunk_blocks,
            max_tokens=chunk_max_tokens,
            overlap_blocks=overlap_blocks,
        )
        extractor = DocumentExtractor(
            backend=extractor_backend,
            provider=extractor_provider,
            granularity=extractor_granularity,
            model=extractor_model,
        )
        extracted = (
            extractor.extract_chunks(semantic_chunks, source_file=str(document_path))
            if parsed_document.markdown.strip()
            else {
                **empty_extracted_payload(),
                "chunk_extractions": [],
                "summary": {"chunk_count": 0, "concept_count": 0, "relation_count": 0},
            }
        )
        extracted_path = extraction_dir / "extracted.json"
        write_json(extracted_path, extracted)

        return DocumentProcessingDomainResult(
            parsed_document=parsed_document,
            extracted=extracted,
            parsed_path=parsed_path,
            extracted_path=extracted_path,
        )
