from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from src.knowledge.schema import EdgeRelation, GraphEdge, GraphNode, MultimodalDocumentGraph, NodeKind


@dataclass
class TopicCandidate:
    id: str
    label: str
    description: str
    seed_source: str
    topic_type: str
    section_paths: list[list[str]] = field(default_factory=list)
    concept_ids: set[str] = field(default_factory=set)
    chunk_ids: set[str] = field(default_factory=set)
    artifact_ids: set[str] = field(default_factory=set)
    seed_concept_ids: set[str] = field(default_factory=set)
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["concept_ids"] = sorted(self.concept_ids)
        payload["chunk_ids"] = sorted(self.chunk_ids)
        payload["artifact_ids"] = sorted(self.artifact_ids)
        payload["seed_concept_ids"] = sorted(self.seed_concept_ids)
        payload["section_paths"] = [list(path) for path in self.section_paths]
        return payload


@dataclass
class TopicInductionResult:
    topic_nodes: list[GraphNode]
    topic_edges: list[GraphEdge]
    topic_candidates: list[TopicCandidate]
    consolidated_topics: list[TopicCandidate]
    accepted_topics: list[TopicCandidate]
    summary: dict[str, Any]

    def checkpoints(self) -> dict[str, Any]:
        return {
            "topic_candidates": {
                "count": len(self.topic_candidates),
                "topics": [topic.to_dict() for topic in self.topic_candidates],
            },
            "topic_consolidation": {
                "count": len(self.consolidated_topics),
                "topics": [topic.to_dict() for topic in self.consolidated_topics],
            },
            "topics": {
                "count": len(self.accepted_topics),
                "topics": [topic.to_dict() for topic in self.accepted_topics],
                "summary": self.summary,
            },
        }


def induce_topics(
    graph: MultimodalDocumentGraph,
    *,
    min_concepts: int = 2,
    min_grounding: int = 2,
    duplicate_threshold: float = 0.6,
) -> TopicInductionResult:
    node_map = {node.id: node for node in graph.nodes}
    concept_nodes = [node for node in graph.nodes if node.kind == NodeKind.concept]
    section_nodes = [node for node in graph.nodes if node.kind == NodeKind.section]
    chunk_nodes = [node for node in graph.nodes if node.kind == NodeKind.chunk]
    artifact_nodes = [node for node in graph.nodes if node.kind == NodeKind.artifact]

    concept_to_chunks: dict[str, set[str]] = defaultdict(set)
    concept_to_artifacts: dict[str, set[str]] = defaultdict(set)
    concept_neighbors: dict[str, set[str]] = defaultdict(set)
    section_to_chunks: dict[str, set[str]] = defaultdict(set)
    section_to_concepts: dict[str, set[str]] = defaultdict(set)
    chunk_to_section: dict[str, str] = {}
    chunk_to_concepts: dict[str, set[str]] = defaultdict(set)

    for edge in graph.edges:
        source_kind = node_map.get(edge.source).kind if edge.source in node_map else None
        target_kind = node_map.get(edge.target).kind if edge.target in node_map else None

        if edge.relation == EdgeRelation.contains and source_kind == NodeKind.section and target_kind == NodeKind.chunk:
            section_to_chunks[edge.source].add(edge.target)
            chunk_to_section[edge.target] = edge.source

        if edge.relation in {EdgeRelation.mentions, EdgeRelation.defines, EdgeRelation.explains, EdgeRelation.supports}:
            if source_kind == NodeKind.chunk and target_kind == NodeKind.concept:
                concept_to_chunks[edge.target].add(edge.source)
                chunk_to_concepts[edge.source].add(edge.target)
            elif source_kind == NodeKind.artifact and target_kind == NodeKind.concept:
                concept_to_artifacts[edge.target].add(edge.source)

        if source_kind == NodeKind.concept and target_kind == NodeKind.concept:
            concept_neighbors[edge.source].add(edge.target)
            concept_neighbors[edge.target].add(edge.source)

    for chunk_id, concepts in chunk_to_concepts.items():
        section_id = chunk_to_section.get(chunk_id)
        if section_id:
            section_to_concepts[section_id].update(concepts)

    topic_candidates: list[TopicCandidate] = []
    section_seed_topics = _build_section_seed_topics(
        graph.document_id,
        section_nodes,
        section_to_concepts,
        concept_to_chunks,
        concept_to_artifacts,
        concept_neighbors,
        node_map,
    )
    topic_candidates.extend(section_seed_topics)
    topic_candidates.extend(
        _build_concept_anchor_topics(
            graph.document_id,
            concept_nodes,
            concept_to_chunks,
            concept_to_artifacts,
            concept_neighbors,
            node_map,
            reserved_labels={topic.label.casefold() for topic in section_seed_topics},
        )
    )

    for candidate in topic_candidates:
        _expand_candidate(candidate, concept_neighbors, concept_to_chunks, concept_to_artifacts, chunk_to_concepts, node_map)
        candidate.scores = _score_candidate(candidate, node_map)

    consolidated = _consolidate_topics(topic_candidates, duplicate_threshold=duplicate_threshold)
    for candidate in consolidated:
        candidate.scores = _score_candidate(candidate, node_map)

    accepted = [
        candidate
        for candidate in consolidated
        if len(candidate.concept_ids) >= min_concepts
        and (len(candidate.chunk_ids) + len(candidate.artifact_ids)) >= min_grounding
        and candidate.scores.get("pedagogical_usefulness", 0.0) >= 0.45
    ]

    topic_nodes, topic_edges = _materialize_topics(
        graph.document_id,
        graph.source_file,
        accepted,
        node_map,
        chunk_to_section,
        section_to_chunks,
    )

    return TopicInductionResult(
        topic_nodes=topic_nodes,
        topic_edges=topic_edges,
        topic_candidates=topic_candidates,
        consolidated_topics=consolidated,
        accepted_topics=accepted,
        summary={
            "candidate_count": len(topic_candidates),
            "consolidated_count": len(consolidated),
            "accepted_count": len(accepted),
        },
    )


def _build_section_seed_topics(
    document_id: str,
    section_nodes: list[GraphNode],
    section_to_concepts: dict[str, set[str]],
    concept_to_chunks: dict[str, set[str]],
    concept_to_artifacts: dict[str, set[str]],
    concept_neighbors: dict[str, set[str]],
    node_map: dict[str, GraphNode],
) -> list[TopicCandidate]:
    topics: list[TopicCandidate] = []
    for index, section in enumerate(section_nodes):
        concept_ids = {
            concept_id
            for concept_id in section_to_concepts.get(section.id, set())
            if _is_topic_worthy_concept(node_map.get(concept_id))
        }
        if len(concept_ids) < 2:
            continue
        label = _normalize_topic_label(section.label)
        seed_concepts = _pick_seed_concepts(concept_ids, concept_to_chunks, concept_to_artifacts, concept_neighbors)
        topics.append(
            TopicCandidate(
                id=f"{document_id}::topic_candidate::section::{index}::{_slug(label)}",
                label=label,
                description=f"Section-seeded topic for {section.label}",
                seed_source="section",
                topic_type="local_topic",
                section_paths=[list(section.section_path)],
                concept_ids=set(concept_ids),
                seed_concept_ids=set(seed_concepts),
                metadata={"section_id": section.id, "source_section_label": section.label},
            )
        )
    return topics


def _build_concept_anchor_topics(
    document_id: str,
    concept_nodes: list[GraphNode],
    concept_to_chunks: dict[str, set[str]],
    concept_to_artifacts: dict[str, set[str]],
    concept_neighbors: dict[str, set[str]],
    node_map: dict[str, GraphNode],
    reserved_labels: set[str],
) -> list[TopicCandidate]:
    topics: list[TopicCandidate] = []
    ranked = sorted(
        concept_nodes,
        key=lambda node: (
            len(concept_to_chunks.get(node.id, set())),
            len(concept_neighbors.get(node.id, set())),
            len(node.metadata.get("definitions", [])),
            node.metadata.get("mention_count", 0),
        ),
        reverse=True,
    )

    added = 0
    for node in ranked:
        if not _is_topic_worthy_concept(node):
            continue
        if len(concept_to_chunks.get(node.id, set())) < 2 and node.metadata.get("mention_count", 0) < 3:
            continue
        concept_ids = {node.id}
        for neighbor_id in concept_neighbors.get(node.id, set()):
            if _is_topic_worthy_concept(node_map.get(neighbor_id)):
                concept_ids.add(neighbor_id)
        if len(concept_ids) < 2:
            continue
        label = _normalize_topic_label(node.label)
        if label.casefold() in reserved_labels:
            continue
        topics.append(
            TopicCandidate(
                id=f"{document_id}::topic_candidate::anchor::{added}::{_slug(node.label)}",
                label=label,
                description=f"Concept-anchored topic centered on {node.label}",
                seed_source="concept_anchor",
                topic_type="local_topic",
                section_paths=[list(node.section_path)] if node.section_path else [],
                concept_ids=concept_ids,
                seed_concept_ids={node.id},
                metadata={"anchor_concept_id": node.id},
            )
        )
        added += 1
        if added >= 12:
            break
    return topics


def _expand_candidate(
    candidate: TopicCandidate,
    concept_neighbors: dict[str, set[str]],
    concept_to_chunks: dict[str, set[str]],
    concept_to_artifacts: dict[str, set[str]],
    chunk_to_concepts: dict[str, set[str]],
    node_map: dict[str, GraphNode],
) -> None:
    base_concepts = set(candidate.concept_ids)
    expanded = set(base_concepts)
    concept_weights: Counter[str] = Counter({concept_id: 4 for concept_id in candidate.seed_concept_ids})
    for concept_id in list(base_concepts):
        concept_weights[concept_id] += 2
        for neighbor_id in concept_neighbors.get(concept_id, set()):
            if _is_topic_worthy_concept(node_map.get(neighbor_id)):
                expanded.add(neighbor_id)
                concept_weights[neighbor_id] += 2
        for chunk_id in concept_to_chunks.get(concept_id, set()):
            related_concepts = {
                related_id
                for related_id in chunk_to_concepts.get(chunk_id, set())
                if _is_topic_worthy_concept(node_map.get(related_id))
            }
            if len(related_concepts) <= 6:
                expanded.update(related_concepts)
                for related_id in related_concepts:
                    concept_weights[related_id] += 1

    scored = []
    for concept_id in expanded:
        node = node_map.get(concept_id)
        if not _is_topic_worthy_concept(node):
            continue
        weight = concept_weights[concept_id]
        weight += min(3, len(concept_to_chunks.get(concept_id, set())))
        weight += min(2, len(concept_to_artifacts.get(concept_id, set())))
        weight += min(2, len(concept_neighbors.get(concept_id, set())) // 2)
        if candidate.section_paths and node and node.section_path:
            if any(node.section_path[: len(path)] == path for path in candidate.section_paths if path):
                weight += 2
        scored.append((weight, concept_id))

    scored.sort(reverse=True)
    max_concepts = 12 if candidate.seed_source == "section" else 10
    candidate.concept_ids = {concept_id for _, concept_id in scored[:max_concepts]}
    for concept_id in candidate.concept_ids:
        candidate.chunk_ids.update(concept_to_chunks.get(concept_id, set()))
        candidate.artifact_ids.update(concept_to_artifacts.get(concept_id, set()))

    chunk_scores: list[tuple[int, str]] = []
    for chunk_id in candidate.chunk_ids:
        overlap = len(chunk_to_concepts.get(chunk_id, set()) & candidate.concept_ids)
        if overlap <= 0:
            continue
        chunk_scores.append((overlap, chunk_id))
    chunk_scores.sort(reverse=True)
    candidate.chunk_ids = {chunk_id for _, chunk_id in chunk_scores[:8]}


def _score_candidate(candidate: TopicCandidate, node_map: dict[str, GraphNode]) -> dict[str, float]:
    concept_labels = [node_map[concept_id].label for concept_id in candidate.concept_ids if concept_id in node_map]
    if not concept_labels:
        return {
            "coherence_score": 0.0,
            "grounding_score": 0.0,
            "breadth_score": 0.0,
            "distinctness_score": 0.0,
            "pedagogical_usefulness": 0.0,
        }

    low_value_count = sum(1 for label in concept_labels if _looks_like_low_value_label(label))
    coherence = max(0.0, 1.0 - (low_value_count / max(len(concept_labels), 1)))
    grounding = min(1.0, (len(candidate.chunk_ids) + len(candidate.artifact_ids)) / max(len(candidate.concept_ids), 1))
    ideal_breadth = 1.0 - abs(len(candidate.concept_ids) - 6) / 10.0
    breadth = max(0.0, min(1.0, ideal_breadth))
    distinctness = 0.8 if len(candidate.seed_concept_ids) >= 1 else 0.5
    if len(candidate.concept_ids) > 14:
        coherence *= 0.7
        distinctness *= 0.8
    pedagogical = min(1.0, (coherence * 0.35) + (grounding * 0.35) + (breadth * 0.3))
    return {
        "coherence_score": round(coherence, 3),
        "grounding_score": round(grounding, 3),
        "breadth_score": round(breadth, 3),
        "distinctness_score": round(distinctness, 3),
        "pedagogical_usefulness": round(pedagogical, 3),
    }


def _consolidate_topics(
    candidates: list[TopicCandidate],
    *,
    duplicate_threshold: float,
) -> list[TopicCandidate]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate.scores.get("pedagogical_usefulness", 0.0),
            len(candidate.concept_ids),
            len(candidate.chunk_ids),
        ),
        reverse=True,
    )
    kept: list[TopicCandidate] = []
    for candidate in ranked:
        merged = False
        for existing in kept:
            overlap = _jaccard(candidate.concept_ids, existing.concept_ids)
            label_similarity = _label_similarity(candidate.label, existing.label)
            if overlap >= duplicate_threshold or (overlap >= 0.4 and label_similarity >= 0.75) or (
                candidate.label.casefold() == existing.label.casefold() and overlap >= 0.15
            ):
                existing.concept_ids.update(candidate.concept_ids)
                existing.chunk_ids.update(candidate.chunk_ids)
                existing.artifact_ids.update(candidate.artifact_ids)
                existing.seed_concept_ids.update(candidate.seed_concept_ids)
                existing.section_paths = _unique_paths(existing.section_paths + candidate.section_paths)
                existing.metadata.setdefault("merged_candidate_ids", []).append(candidate.id)
                if candidate.scores.get("pedagogical_usefulness", 0.0) > existing.scores.get("pedagogical_usefulness", 0.0):
                    existing.label = candidate.label
                    existing.description = candidate.description
                merged = True
                break
        if not merged:
            kept.append(_copy_candidate(candidate))
    return kept


def _materialize_topics(
    document_id: str,
    source_file: str | None,
    topics: list[TopicCandidate],
    node_map: dict[str, GraphNode],
    chunk_to_section: dict[str, str],
    section_to_chunks: dict[str, set[str]],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    topic_nodes: list[GraphNode] = []
    topic_edges: list[GraphEdge] = []
    section_to_topics: dict[str, set[str]] = defaultdict(set)

    for index, topic in enumerate(topics):
        topic_id = f"{document_id}::topic::{index}::{_slug(topic.label)}"
        topic_nodes.append(
            GraphNode(
                id=topic_id,
                label=topic.label,
                kind=NodeKind.topic,
                source_file=source_file,
                section_path=topic.section_paths[0] if topic.section_paths else [],
                metadata={
                    "description": topic.description,
                    "topic_type": topic.topic_type,
                    "seed_source": topic.seed_source,
                    **topic.scores,
                    "seed_concept_ids": sorted(topic.seed_concept_ids),
                    "section_paths": topic.section_paths,
                },
            )
        )

        for concept_id in sorted(topic.concept_ids):
            if concept_id not in node_map:
                continue
            topic_edges.append(
                GraphEdge(
                    source=topic_id,
                    target=concept_id,
                    relation=EdgeRelation.groups,
                    confidence="INFERRED",
                    confidence_score=topic.scores.get("coherence_score", 0.6),
                    source_file=source_file,
                    extraction_method="topic_inducer",
                )
            )

        for chunk_id in sorted(topic.chunk_ids):
            if chunk_id not in node_map:
                continue
            topic_edges.append(
                GraphEdge(
                    source=topic_id,
                    target=chunk_id,
                    relation=EdgeRelation.grounded_by,
                    confidence="INFERRED",
                    confidence_score=topic.scores.get("grounding_score", 0.6),
                    source_file=source_file,
                    source_chunk_id=chunk_id,
                    extraction_method="topic_inducer",
                )
            )
            section_id = chunk_to_section.get(chunk_id)
            if section_id:
                section_to_topics[section_id].add(topic_id)

        for artifact_id in sorted(topic.artifact_ids):
            if artifact_id not in node_map:
                continue
            topic_edges.append(
                GraphEdge(
                    source=topic_id,
                    target=artifact_id,
                    relation=EdgeRelation.illustrated_by,
                    confidence="INFERRED",
                    confidence_score=topic.scores.get("grounding_score", 0.6),
                    source_file=source_file,
                    extraction_method="topic_inducer",
                )
            )

    for section_id, topic_ids in section_to_topics.items():
        for topic_id in sorted(topic_ids):
            topic_edges.append(
                GraphEdge(
                    source=section_id,
                    target=topic_id,
                    relation=EdgeRelation.covers,
                    confidence="INFERRED",
                    confidence_score=0.75,
                    source_file=source_file,
                    extraction_method="topic_inducer",
                )
            )

    return topic_nodes, topic_edges


def _pick_seed_concepts(
    concept_ids: set[str],
    concept_to_chunks: dict[str, set[str]],
    concept_to_artifacts: dict[str, set[str]],
    concept_neighbors: dict[str, set[str]],
) -> list[str]:
    ranked = sorted(
        concept_ids,
        key=lambda concept_id: (
            len(concept_to_chunks.get(concept_id, set())),
            len(concept_neighbors.get(concept_id, set())),
            len(concept_to_artifacts.get(concept_id, set())),
        ),
        reverse=True,
    )
    return ranked[:3]


def _normalize_topic_label(label: str) -> str:
    cleaned = re.sub(r"^\d+(?:\.\d+)*\s*", "", label).strip()
    code_section = bool(re.match(r"^code:\s*", cleaned, flags=re.IGNORECASE))
    cleaned = re.sub(r"^code:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.replace("xv6", "xv6").strip()
    if code_section and cleaned:
        cleaned = f"{cleaned} implementation"
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned or label.strip()


def _is_topic_worthy_concept(node: GraphNode | None) -> bool:
    if node is None or node.kind != NodeKind.concept:
        return False
    label = node.label.strip()
    if not label:
        return False
    if _looks_like_low_value_label(label):
        return False
    return True


def _looks_like_low_value_label(value: str) -> bool:
    label = value.strip()
    normalized = label.casefold()
    if normalized in {"ra", "sp", "a0", "a1", "s0", "s11", "sd", "ld", "ret", "tp"}:
        return True
    if normalized.startswith(("p->", "c->")):
        return True
    if re.fullmatch(r"[asft]\d{1,2}", normalized):
        return True
    if re.fullmatch(r"[a-z]{1,3}", normalized) and len(normalized) <= 3:
        return True
    return False


def _copy_candidate(candidate: TopicCandidate) -> TopicCandidate:
    return TopicCandidate(
        id=candidate.id,
        label=candidate.label,
        description=candidate.description,
        seed_source=candidate.seed_source,
        topic_type=candidate.topic_type,
        section_paths=[list(path) for path in candidate.section_paths],
        concept_ids=set(candidate.concept_ids),
        chunk_ids=set(candidate.chunk_ids),
        artifact_ids=set(candidate.artifact_ids),
        seed_concept_ids=set(candidate.seed_concept_ids),
        scores=dict(candidate.scores),
        metadata=dict(candidate.metadata),
    )


def _unique_paths(paths: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    unique: list[list[str]] = []
    for path in paths:
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(list(path))
    return unique


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _label_similarity(left: str, right: str) -> float:
    left_tokens = set(_slug(left).split("_"))
    right_tokens = set(_slug(right).split("_"))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_") or "topic"
