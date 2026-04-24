from __future__ import annotations

from src.knowledge.schema import EdgeRelation, GraphEdge, MultimodalDocumentGraph


RELATION_MAP = {
    "contains": EdgeRelation.contains,
    "follows": EdgeRelation.follows,
    "mentions": EdgeRelation.mentions,
    "references": EdgeRelation.references,
    "explains": EdgeRelation.explains,
    "illustrates": EdgeRelation.illustrates,
    "supports": EdgeRelation.supports,
    "defines": EdgeRelation.defines,
    "related_to": EdgeRelation.related_to,
    "depends_on": EdgeRelation.depends_on,
    "part_of": EdgeRelation.part_of,
    "causes": EdgeRelation.causes,
    "semantically_similar_to": EdgeRelation.semantically_similar_to,
    "is_a": EdgeRelation.related_to,
    "uses": EdgeRelation.related_to,
}


def consolidate_graph_schema(graph: MultimodalDocumentGraph) -> MultimodalDocumentGraph:
    consolidated = graph.model_copy(deep=True)
    consolidated.edges = [_normalize_edge(edge) for edge in consolidated.edges]
    consolidated.metadata = dict(consolidated.metadata)
    consolidated.metadata["schema_consolidated"] = True
    return consolidated


def _normalize_edge(edge: GraphEdge) -> GraphEdge:
    edge = edge.model_copy(deep=True)
    original = edge.relation.value
    edge.relation = RELATION_MAP.get(original, EdgeRelation.related_to)
    if edge.relation.value != original:
        edge.metadata = dict(edge.metadata)
        edge.metadata["original_relation"] = original
    return edge
