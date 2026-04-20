from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None  # type: ignore[assignment]

from src.document_understanding.extractor import DocumentExtractor
from src.document_understanding.parser import parse_document
from src.knowledge.kg_builder import build_knowledge_graph
from src.utils.io import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class QuizGenerationPipeline:
    """End-to-end pipeline from document understanding to verification."""

    def run(self, document_path: str | Path, *, output_path: str | Path) -> dict[str, object]:
        document_path = Path(document_path)
        parsed_document = parse_document(document_path)
        extractor_backend = os.getenv("QUIZGEN_EXTRACTOR_BACKEND", "langchain")
        extractor_provider = os.getenv("QUIZGEN_LLM_PROVIDER", "openai")
        extractor_granularity = os.getenv("QUIZGEN_EXTRACTION_GRANULARITY", "balanced")
        extractor_model = os.getenv("QUIZGEN_LLM_MODEL")
        extractor = DocumentExtractor(
            backend=extractor_backend,
            provider=extractor_provider,
            granularity=extractor_granularity,
            model=extractor_model,
        )

        extracted = extractor.extract(parsed_document.markdown) if parsed_document.markdown.strip() else {
            "concepts": [],
            "definitions": {},
            "relations": [],
            "examples": [],
        }

        document_graph = build_knowledge_graph(
            {
                "markdown": parsed_document.markdown,
                "sections": parsed_document.sections,
                "paragraphs": parsed_document.paragraphs,
                "figures": parsed_document.figures,
                "captions": parsed_document.captions,
            },
            extracted,
            source_file=document_path,
        )

        result = {
            "document": {
                "sections": parsed_document.sections,
                "paragraphs": parsed_document.paragraphs,
                "figures": parsed_document.figures,
                "captions": parsed_document.captions,
            },
            "extracted": extracted,
            "graph": document_graph.model_dump(mode="json"),
            "graph_summary": document_graph.summary(),
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_file, result)
        return result
