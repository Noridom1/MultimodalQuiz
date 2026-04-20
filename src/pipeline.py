from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.document_understanding.extractor import DocumentExtractor
from src.document_understanding.parser import parse_document
from src.utils.io import write_json


@dataclass
class QuizGenerationPipeline:
    """End-to-end pipeline from document understanding to verification."""

    def run(self, document_path: str | Path, *, output_path: str | Path) -> dict[str, object]:
        parsed_document = parse_document(document_path)
        extractor = DocumentExtractor()

        combined_text = "\n\n".join(parsed_document.paragraphs)
        extracted = extractor.extract(combined_text) if combined_text.strip() else {
            "concepts": [],
            "definitions": {},
            "relations": [],
            "examples": [],
        }

        result = {
            "document": {
                "sections": parsed_document.sections,
                "paragraphs": parsed_document.paragraphs,
                "figures": parsed_document.figures,
                "captions": parsed_document.captions,
            },
            "extracted": extracted,
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_file, result)
        return result
