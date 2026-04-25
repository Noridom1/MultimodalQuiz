from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.knowledge.schema import EdgeRelation, MultimodalDocumentGraph, NodeKind


SEMANTIC_RELATIONS = {
    EdgeRelation.mentions,
    EdgeRelation.explains,
    EdgeRelation.defines,
    EdgeRelation.illustrates,
    EdgeRelation.supports,
    EdgeRelation.related_to,
    EdgeRelation.depends_on,
    EdgeRelation.part_of,
    EdgeRelation.causes,
    EdgeRelation.semantically_similar_to,
}


@dataclass
class GraphValidationReport:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_graph(graph: MultimodalDocumentGraph) -> GraphValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    node_ids = {node.id for node in graph.nodes}
    nodes_by_kind: dict[NodeKind, list[str]] = {}
    for node in graph.nodes:
        nodes_by_kind.setdefault(node.kind, []).append(node.id)

    document_nodes = nodes_by_kind.get(NodeKind.document, [])
    if len(document_nodes) != 1:
        errors.append(f"expected exactly 1 document node, found {len(document_nodes)}")

    chunk_parent_counts: dict[str, int] = {}
    grounded_concepts: set[str] = set()
    topic_to_concepts: dict[str, set[str]] = {}
    topic_grounding_counts: dict[str, int] = {}
    topic_labels: dict[str, str] = {}

    for edge in graph.edges:
        if edge.source not in node_ids:
            errors.append(f"edge source {edge.source} does not exist")
        if edge.target not in node_ids:
            errors.append(f"edge target {edge.target} does not exist")
        if edge.relation == EdgeRelation.contains and edge.target in nodes_by_kind.get(NodeKind.chunk, []):
            chunk_parent_counts[edge.target] = chunk_parent_counts.get(edge.target, 0) + 1
        if edge.relation in {EdgeRelation.mentions, EdgeRelation.explains, EdgeRelation.defines}:
            grounded_concepts.add(edge.target)
        if edge.relation == EdgeRelation.groups:
            topic_to_concepts.setdefault(edge.source, set()).add(edge.target)
        if edge.relation in {EdgeRelation.grounded_by, EdgeRelation.illustrated_by}:
            topic_grounding_counts[edge.source] = topic_grounding_counts.get(edge.source, 0) + 1
        if edge.relation in SEMANTIC_RELATIONS:
            if not edge.source_chunk_id:
                errors.append(f"semantic edge {edge.source}->{edge.target} lacks source_chunk_id")
            if edge.confidence not in {"EXTRACTED", "INFERRED", "AMBIGUOUS"}:
                errors.append(f"semantic edge {edge.source}->{edge.target} has invalid confidence")

    for chunk_id in nodes_by_kind.get(NodeKind.chunk, []):
        parent_count = chunk_parent_counts.get(chunk_id, 0)
        if parent_count != 1:
            errors.append(f"chunk {chunk_id} has {parent_count} parent sections")

    for concept_id in nodes_by_kind.get(NodeKind.concept, []):
        if concept_id not in grounded_concepts:
            warnings.append(f"concept {concept_id} has no chunk grounding edge")

    for topic_id in nodes_by_kind.get(NodeKind.topic, []):
        concept_count = len(topic_to_concepts.get(topic_id, set()))
        grounding_count = topic_grounding_counts.get(topic_id, 0)
        topic_node = next((node for node in graph.nodes if node.id == topic_id), None)
        label = topic_node.label if topic_node is not None else topic_id
        topic_labels[topic_id] = label
        if concept_count < 2:
            errors.append(f"topic {topic_id} groups only {concept_count} concepts")
        if grounding_count < 2:
            errors.append(f"topic {topic_id} has only {grounding_count} grounding edges")
        if not _is_human_readable_topic_label(label):
            errors.append(f"topic {topic_id} has non-human-readable label: {label}")

    topic_ids = nodes_by_kind.get(NodeKind.topic, [])
    for index, left_id in enumerate(topic_ids):
        for right_id in topic_ids[index + 1 :]:
            overlap = _jaccard(topic_to_concepts.get(left_id, set()), topic_to_concepts.get(right_id, set()))
            if overlap >= 0.85:
                warnings.append(
                    f"topics {topic_labels.get(left_id, left_id)} and {topic_labels.get(right_id, right_id)} have high concept overlap ({overlap:.2f})"
                )

    stats = {
        "document_nodes": len(document_nodes),
        "section_nodes": len(nodes_by_kind.get(NodeKind.section, [])),
        "chunk_nodes": len(nodes_by_kind.get(NodeKind.chunk, [])),
        "concept_nodes": len(nodes_by_kind.get(NodeKind.concept, [])),
        "topic_nodes": len(nodes_by_kind.get(NodeKind.topic, [])),
        "artifact_nodes": len(nodes_by_kind.get(NodeKind.artifact, [])),
        "edge_count": len(graph.edges),
    }
    return GraphValidationReport(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def _is_human_readable_topic_label(label: str) -> bool:
    text = label.strip()
    if len(text) < 4:
        return False
    if text.lower() in {"topic", "concept", "section"}:
        return False
    return True


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
