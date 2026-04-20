from __future__ import annotations

import re

from src.document_understanding.chunking import semantic_chunk
from src.document_understanding.normalizer import normalize_concepts


class DocumentExtractor:
    """Extract concepts, definitions, relations, and examples from text."""

    def extract(self, text: str) -> dict[str, object]:
        sentences = _split_sentences(text)
        chunks = semantic_chunk(sentences)

        concepts: list[str] = []
        definitions: dict[str, str] = {}
        relations: list[dict[str, str]] = []
        examples: list[str] = []

        for chunk in chunks:
            chunk_text = " ".join(chunk)
            concepts.extend(_extract_candidate_concepts(chunk_text))
            examples.extend(_extract_examples(chunk_text))
            definition_pairs = _extract_definitions(chunk_text)
            for concept, definition in definition_pairs:
                definitions[concept] = definition
            relations.extend(_extract_relations(chunk_text))

        normalized_concepts = normalize_concepts(concepts)
        definitions = {
            key: value
            for key, value in definitions.items()
            if key in normalized_concepts
        }

        return {
            "concepts": normalized_concepts,
            "definitions": definitions,
            "relations": relations,
            "examples": list(dict.fromkeys(example.strip() for example in examples if example.strip())),
        }


def _split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [piece.strip() for piece in pieces if piece and piece.strip()]


def _extract_candidate_concepts(text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}|[A-Za-z][A-Za-z0-9_-]{4,})\b", text):
        candidate = match.group(1).strip()
        if len(candidate) < 3:
            continue
        if candidate.lower() in {"this", "that", "these", "those", "figure", "table", "section"}:
            continue
        candidates.append(candidate)
    return candidates


def _extract_definitions(text: str) -> list[tuple[str, str]]:
    patterns = [
        re.compile(r"(?P<concept>[A-Z][A-Za-z0-9_\- ]{2,40})\s+(?:is|are|refers to|means)\s+(?P<definition>[^.]+)", re.IGNORECASE),
        re.compile(r"(?P<concept>[A-Z][A-Za-z0-9_\- ]{2,40})\s*:\s*(?P<definition>[^.]+)", re.IGNORECASE),
    ]
    pairs: list[tuple[str, str]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            concept = match.group("concept").strip()
            definition = match.group("definition").strip()
            if concept and definition:
                pairs.append((concept, definition))
    return pairs


def _extract_relations(text: str) -> list[dict[str, str]]:
    relations: list[dict[str, str]] = []
    relation_patterns = [
        ("causes", re.compile(r"(?P<src>[A-Z][A-Za-z0-9_\- ]{2,40})\s+causes\s+(?P<tgt>[A-Z][A-Za-z0-9_\- ]{2,40})", re.IGNORECASE)),
        ("depends_on", re.compile(r"(?P<src>[A-Z][A-Za-z0-9_\- ]{2,40})\s+depends on\s+(?P<tgt>[A-Z][A-Za-z0-9_\- ]{2,40})", re.IGNORECASE)),
        ("part_of", re.compile(r"(?P<src>[A-Z][A-Za-z0-9_\- ]{2,40})\s+is part of\s+(?P<tgt>[A-Z][A-Za-z0-9_\- ]{2,40})", re.IGNORECASE)),
    ]
    for relation_name, pattern in relation_patterns:
        for match in pattern.finditer(text):
            relations.append(
                {
                    "source": match.group("src").strip(),
                    "target": match.group("tgt").strip(),
                    "relation": relation_name,
                }
            )
    return relations


def _extract_examples(text: str) -> list[str]:
    examples: list[str] = []
    for match in re.finditer(r"(?:for example|for instance|e\.g\.)[:\s]+(?P<example>[^.]+)", text, re.IGNORECASE):
        examples.append(match.group("example").strip())
    return examples
