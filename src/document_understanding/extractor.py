from __future__ import annotations

import logging
import os
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.document_understanding.chunking import SemanticChunk, semantic_chunk
from src.document_understanding.normalizer import normalize_concepts


logger = logging.getLogger(__name__)


ExtractorBackend = Literal["rule", "langchain"]
LLMProvider = Literal["openai", "mistral"]
ExtractionGranularity = Literal["coarse", "balanced", "fine"]


class DefinitionItem(BaseModel):
    concept: str
    definition: str


class RelationItem(BaseModel):
    source: str
    target: str
    relation: str
    confidence: str = "EXTRACTED"
    confidence_score: float = 1.0


class ChunkExtraction(BaseModel):
    chunk_index: int
    chunk_id: str | None = None
    source_file: str | None = None
    section_path: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    definitions: list[DefinitionItem] = Field(default_factory=list)
    relations: list[RelationItem] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    extraction_method: str = "rule"

    @field_validator("examples", mode="before")
    @classmethod
    def _normalize_examples(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            normalized: list[str] = []
            for item in value:
                if isinstance(item, str):
                    text = item.strip()
                elif isinstance(item, dict):
                    raw = item.get("example") or item.get("text") or item.get("value")
                    text = str(raw).strip() if raw is not None else ""
                else:
                    text = str(item).strip()
                if text:
                    normalized.append(text)
            return normalized
        text = str(value).strip()
        return [text] if text else []


class BatchExtraction(BaseModel):
    chunks: list[ChunkExtraction] = Field(default_factory=list)


class DocumentExtractor:
    """Extract concepts, definitions, relations, and examples from text or chunks."""

    def __init__(
        self,
        *,
        backend: ExtractorBackend = "rule",
        provider: LLMProvider = "mistral",
        granularity: ExtractionGranularity = "balanced",
        model: str | None = None,
        batch_size: int = 4,
        max_calls: int = 24,
    ) -> None:
        self.backend = backend
        self.provider = provider
        self.granularity = _normalize_granularity(granularity)
        self.model = model or _default_model_for_provider(provider)
        self.batch_size = max(1, batch_size)
        self.max_calls = max(1, max_calls)

    def extract(self, text: str) -> dict[str, object]:
        sentences = _split_sentences(text)
        chunks = semantic_chunk(sentences)
        if self.backend == "langchain":
            try:
                return self._extract_with_langchain(chunks)
            except Exception as exc:
                logger.warning("LangChain extraction failed, falling back to rule-based extraction: %s", exc)
        return self._extract_with_rules(chunks)

    def extract_chunks(self, chunks: list[SemanticChunk], *, source_file: str | None = None) -> dict[str, object]:
        if self.backend == "langchain":
            try:
                chunk_extractions = self._extract_chunk_objects_with_langchain(chunks, source_file=source_file)
            except Exception as exc:
                logger.warning("LangChain chunk extraction failed, falling back to rule-based extraction: %s", exc)
                chunk_extractions = self._extract_chunk_objects_with_rules(chunks, source_file=source_file)
        else:
            chunk_extractions = self._extract_chunk_objects_with_rules(chunks, source_file=source_file)

        return self._finalize_chunk_extractions(chunk_extractions)

    def _extract_with_rules(self, chunks: list[list[str]]) -> dict[str, object]:
        concepts: list[str] = []
        definitions: dict[str, str] = {}
        relations: list[dict[str, object]] = []
        examples: list[str] = []

        for chunk in chunks:
            chunk_text = " ".join(chunk)
            concepts.extend(_extract_candidate_concepts(chunk_text))
            examples.extend(_extract_examples(chunk_text))
            for concept, definition in _extract_definitions(chunk_text):
                definitions[concept] = definition
            relations.extend(_extract_relations(chunk_text))

        normalized_concepts = normalize_concepts(concepts)
        definitions = {key: value for key, value in definitions.items() if key in normalized_concepts}
        return self._apply_granularity(
            {
                "concepts": normalized_concepts,
                "definitions": definitions,
                "relations": relations,
                "examples": list(dict.fromkeys(example.strip() for example in examples if example.strip())),
            }
        )

    def _extract_chunk_objects_with_rules(
        self,
        chunks: list[SemanticChunk],
        *,
        source_file: str | None = None,
    ) -> list[ChunkExtraction]:
        chunk_extractions: list[ChunkExtraction] = []
        for index, chunk in enumerate(chunks):
            concepts = _extract_candidate_concepts(chunk.text)
            definitions = [DefinitionItem(concept=concept, definition=definition) for concept, definition in _extract_definitions(chunk.text)]
            relations = [RelationItem(**relation) for relation in _extract_relations(chunk.text)]
            examples = _extract_examples(chunk.text)
            chunk_extractions.append(
                ChunkExtraction(
                    chunk_index=index,
                    chunk_id=chunk.id,
                    source_file=source_file,
                    section_path=list(chunk.section_path),
                    concepts=concepts,
                    definitions=definitions,
                    relations=relations,
                    examples=examples,
                    extraction_method="rule",
                )
            )
        return chunk_extractions

    def _extract_with_langchain(self, chunks: list[list[str]]) -> dict[str, object]:
        if not chunks:
            return {"concepts": [], "definitions": {}, "relations": [], "examples": []}

        chunk_extractions = self._invoke_langchain_batches(
            [{"chunk_index": index, "text": " ".join(chunk)} for index, chunk in enumerate(chunks)]
        )
        return self._finalize_chunk_extractions(chunk_extractions)

    def _extract_chunk_objects_with_langchain(
        self,
        chunks: list[SemanticChunk],
        *,
        source_file: str | None = None,
    ) -> list[ChunkExtraction]:
        payloads = [
            {
                "chunk_index": index,
                "chunk_id": chunk.id,
                "text": chunk.text,
                "section_path": chunk.section_path,
                "source_file": source_file,
            }
            for index, chunk in enumerate(chunks)
        ]
        chunk_extractions = self._invoke_langchain_batches(payloads)
        for item in chunk_extractions:
            item.source_file = source_file
        return chunk_extractions

    def _invoke_langchain_batches(self, payloads: list[dict[str, object]]) -> list[ChunkExtraction]:
        if not payloads:
            return []

        try:
            from langchain_core.prompts import ChatPromptTemplate  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("LangChain dependencies are missing. Install 'langchain' and 'langchain-core'.") from exc

        llm = _build_langchain_chat_model(provider=self.provider, model=self.model)
        structured_llm = llm.with_structured_output(BatchExtraction)
        granularity_instructions = _granularity_instructions(self.granularity)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You extract semantic knowledge for a graph-building pipeline. "
                    "Return concise, high-precision outputs only. " + granularity_instructions,
                ),
                (
                    "user",
                    "Extract from each chunk: concepts, definitions, relations, and examples.\n"
                    "Rules:\n"
                    "- Concepts: technical terms and named concepts only.\n"
                    "- Definitions: include only explicit definitions in the chunk.\n"
                    "- Relations: use normalized relation names such as causes, depends_on, part_of, related_to.\n"
                    "- Examples: short spans introduced by 'for example', 'for instance', or equivalent.\n"
                    "- Be conservative; skip uncertain items.\n"
                    "\nChunk payload:\n{chunk_payload}",
                ),
            ]
        )

        extracted_rows: list[ChunkExtraction] = []
        for batch_index, batch in enumerate(_batch_payloads(payloads, self.batch_size)):
            if batch_index >= self.max_calls:
                break
            chain = prompt | structured_llm
            response = chain.invoke({"chunk_payload": batch})
            for item in response.chunks:
                source = _find_payload(batch, item.chunk_index)
                item.chunk_id = str(source.get("chunk_id")) if source.get("chunk_id") else item.chunk_id
                item.section_path = list(source.get("section_path", []))
                item.source_file = str(source.get("source_file")) if source.get("source_file") else item.source_file
                item.extraction_method = "langchain"
                extracted_rows.append(item)
        return extracted_rows

    def _finalize_chunk_extractions(self, chunk_extractions: list[ChunkExtraction]) -> dict[str, object]:
        concepts: list[str] = []
        definitions: dict[str, str] = {}
        relations: list[dict[str, object]] = []
        examples: list[str] = []

        normalized_rows: list[dict[str, object]] = []
        for row in chunk_extractions:
            concepts.extend(row.concepts)
            examples.extend(row.examples)
            for definition in row.definitions:
                if definition.concept.strip() and definition.definition.strip():
                    definitions[definition.concept.strip()] = definition.definition.strip()
            for relation in row.relations:
                if relation.source.strip() and relation.target.strip():
                    relations.append(
                        {
                            "source": relation.source.strip(),
                            "target": relation.target.strip(),
                            "relation": relation.relation.strip() or "related_to",
                            "confidence": relation.confidence,
                            "confidence_score": relation.confidence_score,
                            "source_chunk_id": row.chunk_id,
                            "source_file": row.source_file,
                            "extraction_method": row.extraction_method,
                        }
                    )
            normalized_rows.append(row.model_dump(mode="json"))

        normalized_concepts = normalize_concepts(concepts)
        definitions = {key: value for key, value in definitions.items() if key in normalized_concepts}
        payload = self._apply_granularity(
            {
                "concepts": normalized_concepts,
                "definitions": definitions,
                "relations": relations,
                "examples": list(dict.fromkeys(example.strip() for example in examples if example.strip())),
            }
        )
        payload["chunk_extractions"] = normalized_rows
        payload["summary"] = {
            "chunk_count": len(chunk_extractions),
            "concept_count": len(payload["concepts"]),
            "relation_count": len(payload["relations"]),
        }
        return payload

    def _apply_granularity(self, extracted: dict[str, object]) -> dict[str, object]:
        concepts = [str(concept).strip() for concept in extracted.get("concepts", []) if str(concept).strip()]
        definitions = {
            str(key).strip(): str(value).strip()
            for key, value in dict(extracted.get("definitions", {})).items()
            if str(key).strip() and str(value).strip()
        }
        relations = [relation for relation in list(extracted.get("relations", [])) if isinstance(relation, dict)]
        examples = [str(example).strip() for example in extracted.get("examples", []) if str(example).strip()]

        if self.granularity == "fine":
            return {
                "concepts": list(dict.fromkeys(concepts)),
                "definitions": definitions,
                "relations": relations,
                "examples": list(dict.fromkeys(examples)),
            }

        if self.granularity == "coarse":
            concepts = [concept for concept in concepts if _is_coarse_concept(concept)]
            allowed = {concept.casefold() for concept in concepts}
            definitions = {key: value for key, value in definitions.items() if key.casefold() in allowed}
            relations = [
                relation
                for relation in relations
                if str(relation.get("source", "")).strip().casefold() in allowed
                and str(relation.get("target", "")).strip().casefold() in allowed
            ]

        return {
            "concepts": list(dict.fromkeys(concepts)),
            "definitions": definitions,
            "relations": relations,
            "examples": list(dict.fromkeys(examples)),
        }


def _batch_payloads(payloads: list[dict[str, object]], batch_size: int) -> list[list[dict[str, object]]]:
    return [payloads[i : i + batch_size] for i in range(0, len(payloads), batch_size)]


def _normalize_granularity(granularity: str) -> ExtractionGranularity:
    candidate = granularity.strip().lower()
    if candidate not in {"coarse", "balanced", "fine"}:
        return "balanced"
    return candidate  # type: ignore[return-value]


def _granularity_instructions(granularity: ExtractionGranularity) -> str:
    if granularity == "coarse":
        return "Prefer chapter-level concepts. Avoid code symbols, register names, field names, and function internals unless they are central."
    if granularity == "fine":
        return "Include implementation-level concepts, code symbols, registers, state fields, and operational details when helpful."
    return "Balance high-level concepts with key implementation terms, but avoid noisy symbols and redundant low-value details."


def _is_coarse_concept(concept: str) -> bool:
    text = concept.strip()
    if not text or len(text) <= 2:
        return False
    if any(symbol in text for symbol in ("->", "()", "::", "[", "]", "{", "}", "/")):
        return False
    if re.fullmatch(r"[a-z_][a-z0-9_]*", text):
        return False
    return True


def _default_model_for_provider(provider: LLMProvider) -> str:
    env_model = os.getenv("QUIZGEN_LLM_MODEL")
    if env_model:
        return env_model
    if provider == "mistral":
        return "devstral-medium-latest"
    return "gpt-4o-mini"


def _build_langchain_chat_model(*, provider: LLMProvider, model: str):
    if provider == "mistral":
        try:
            from langchain_mistralai import ChatMistralAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Mistral provider selected but dependency is missing. Install 'langchain-mistralai'.") from exc
        return ChatMistralAI(model=model, temperature=0)

    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenAI provider selected but dependency is missing. Install 'langchain-openai'.") from exc
    return ChatOpenAI(model=model, temperature=0)


def _find_payload(batch: list[dict[str, object]], chunk_index: int) -> dict[str, object]:
    for item in batch:
        if int(item.get("chunk_index", -1)) == chunk_index:
            return item
    return {}


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


def _extract_relations(text: str) -> list[dict[str, object]]:
    relations: list[dict[str, object]] = []
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
                    "confidence": "EXTRACTED",
                    "confidence_score": 1.0,
                }
            )
    return relations


def _extract_examples(text: str) -> list[str]:
    examples: list[str] = []
    for match in re.finditer(r"(?:for example|for instance|e\.g\.)[:\s]+(?P<example>[^.]+)", text, re.IGNORECASE):
        examples.append(match.group("example").strip())
    return examples
