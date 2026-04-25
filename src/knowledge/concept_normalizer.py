from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConceptMention:
    label: str
    chunk_id: str | None = None
    source_file: str | None = None
    section_path: list[str] = field(default_factory=list)
    definition: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalConcept:
    id: str
    label: str
    normalized_label: str
    aliases: list[str] = field(default_factory=list)
    mention_count: int = 0
    definitions: list[str] = field(default_factory=list)
    source_chunk_ids: list[str] = field(default_factory=list)
    section_paths: list[list[str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateMerge:
    concept_ids: list[str]
    labels: list[str]
    score: float
    reason: str


@dataclass
class ConceptNormalizationResult:
    canonical_concepts: list[CanonicalConcept]
    mention_to_concept: dict[str, str]
    candidate_merges: list[CandidateMerge]
    alias_map: dict[str, str]
    summary: dict[str, Any]


def normalize_concept_mentions(
    mentions: list[ConceptMention],
    *,
    document_id: str,
    auto_merge_threshold: float = 0.9,
    review_threshold: float = 0.75,
) -> ConceptNormalizationResult:
    canonical_by_id: dict[str, CanonicalConcept] = {}
    canonical_order: list[str] = []
    mention_to_concept: dict[str, str] = {}
    candidate_merges: dict[tuple[str, str], CandidateMerge] = {}
    alias_map: dict[str, str] = {}

    for mention in mentions:
        cleaned = _normalize_surface(mention.label)
        if not cleaned:
            continue

        best_id, best_score, best_reason = _best_candidate(cleaned, mention.label, canonical_by_id)
        mention_key = _mention_key(mention)

        if best_id is not None and best_score >= auto_merge_threshold:
            canonical = canonical_by_id[best_id]
            _attach_mention(canonical, mention)
            mention_to_concept[mention_key] = canonical.id
            alias_map[mention.label] = canonical.id
            continue

        concept_id = _make_concept_id(document_id, mention.label, len(canonical_order))
        canonical = CanonicalConcept(
            id=concept_id,
            label=mention.label.strip(),
            normalized_label=cleaned,
        )
        _attach_mention(canonical, mention)
        canonical_by_id[concept_id] = canonical
        canonical_order.append(concept_id)
        mention_to_concept[mention_key] = concept_id
        alias_map[mention.label] = concept_id

        if best_id is not None and best_score >= review_threshold:
            key = tuple(sorted((best_id, concept_id)))
            candidate_merges[key] = CandidateMerge(
                concept_ids=[best_id, concept_id],
                labels=[canonical_by_id[best_id].label, canonical.label],
                score=best_score,
                reason=best_reason,
            )

    _refresh_labels(canonical_by_id)
    return ConceptNormalizationResult(
        canonical_concepts=[canonical_by_id[concept_id] for concept_id in canonical_order],
        mention_to_concept=mention_to_concept,
        candidate_merges=sorted(candidate_merges.values(), key=lambda item: item.score, reverse=True),
        alias_map=alias_map,
        summary={
            "mention_count": len(mentions),
            "canonical_concept_count": len(canonical_order),
            "candidate_merge_count": len(candidate_merges),
        },
    )


def _attach_mention(canonical: CanonicalConcept, mention: ConceptMention) -> None:
    label = mention.label.strip()
    if label and label not in canonical.aliases:
        canonical.aliases.append(label)
    canonical.mention_count += 1
    if mention.definition:
        definition = mention.definition.strip()
        if definition and definition not in canonical.definitions:
            canonical.definitions.append(definition)
    if mention.chunk_id and mention.chunk_id not in canonical.source_chunk_ids:
        canonical.source_chunk_ids.append(mention.chunk_id)
    if mention.section_path and mention.section_path not in canonical.section_paths:
        canonical.section_paths.append(list(mention.section_path))
    if mention.metadata:
        canonical.metadata.setdefault("supporting_mentions", []).append(dict(mention.metadata))


def _refresh_labels(canonical_by_id: dict[str, CanonicalConcept]) -> None:
    for canonical in canonical_by_id.values():
        if not canonical.aliases:
            continue
        canonical.label = _pick_canonical_label(canonical.aliases)
        canonical.aliases = sorted(dict.fromkeys(alias for alias in canonical.aliases if alias != canonical.label))


def _pick_canonical_label(labels: list[str]) -> str:
    counts = Counter(label.strip() for label in labels if label and label.strip())
    if not counts:
        return ""
    ranked = sorted(
        counts.items(),
        key=lambda item: (
            _looks_like_long_form(item[0]),
            len(item[0]),
            item[1],
        ),
        reverse=True,
    )
    return ranked[0][0]


def _best_candidate(
    cleaned: str,
    raw_label: str,
    canonical_by_id: dict[str, CanonicalConcept],
) -> tuple[str | None, float, str]:
    best_id: str | None = None
    best_score = 0.0
    best_reason = ""
    for canonical in canonical_by_id.values():
        score, reason = _candidate_score(cleaned, raw_label, canonical)
        if score > best_score:
            best_id = canonical.id
            best_score = score
            best_reason = reason
    return best_id, best_score, best_reason


def _candidate_score(cleaned: str, raw_label: str, canonical: CanonicalConcept) -> tuple[float, str]:
    if cleaned == canonical.normalized_label:
        return 1.0, "exact_normalized_match"

    canonical_tokens = set(canonical.normalized_label.split())
    cleaned_tokens = set(cleaned.split())
    if canonical_tokens and cleaned_tokens and canonical_tokens == cleaned_tokens:
        return 0.94, "token_equivalent"

    if _acronym(raw_label) and _acronym(raw_label) == _acronym(canonical.label):
        return 0.92, "acronym_match"

    if cleaned.replace("-", " ") == canonical.normalized_label.replace("-", " "):
        return 0.9, "hyphenation_variant"

    overlap = len(canonical_tokens & cleaned_tokens)
    if overlap and overlap == min(len(canonical_tokens), len(cleaned_tokens)):
        return 0.82, "subset_token_overlap"

    return 0.0, ""


def _mention_key(mention: ConceptMention) -> str:
    return f"{mention.chunk_id or 'global'}::{_normalize_surface(mention.label)}"


def _normalize_surface(text: str) -> str:
    normalized = text.casefold()
    normalized = re.sub(r"\([^)]*\)", " ", normalized)
    normalized = re.sub(r"[_/]", " ", normalized)
    normalized = re.sub(r"[^a-z0-9+\- ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" -")
    return normalized


def _acronym(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text)
    if len(words) <= 1:
        compact = re.sub(r"[^A-Za-z0-9]+", "", text)
        if compact.isupper() and len(compact) > 1:
            return compact
        return ""
    return "".join(word[0].upper() for word in words if word)


def _looks_like_long_form(label: str) -> bool:
    return len(label.split()) >= 2 or any(ch in label for ch in {"-", " "})


def _make_concept_id(document_id: str, label: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")[:48] or "concept"
    return f"{document_id}::concept::{index}::{slug}"
