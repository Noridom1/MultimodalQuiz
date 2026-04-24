from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.knowledge.graph_reviewer import GraphReviewReport, MergeProposal
from src.knowledge.schema import GraphEdge, GraphNode, MultimodalDocumentGraph, NodeKind


@dataclass
class MergeApplicationReport:
    applied_merges: list[dict[str, Any]] = field(default_factory=list)
    skipped_merges: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied_merges": self.applied_merges,
            "skipped_merges": self.skipped_merges,
        }


def apply_merge_proposals(
    graph: MultimodalDocumentGraph,
    review: GraphReviewReport,
    *,
    auto_merge_threshold: float = 0.9,
    assisted_merge_threshold: float = 0.8,
) -> tuple[MultimodalDocumentGraph, MergeApplicationReport]:
    node_map = {node.id: node.model_copy(deep=True) for node in graph.nodes}
    edges = [edge.model_copy(deep=True) for edge in graph.edges]
    report = MergeApplicationReport()

    id_redirects: dict[str, str] = {}

    for proposal in review.merge_proposals:
        accepted, reason = _should_apply(proposal, auto_merge_threshold, assisted_merge_threshold)
        if not accepted:
            report.skipped_merges.append(
                {
                    "proposal": asdict(proposal),
                    "reason": reason,
                }
            )
            continue

        target_id = _choose_target_id(proposal, node_map)
        if target_id is None:
            report.skipped_merges.append(
                {
                    "proposal": asdict(proposal),
                    "reason": "missing_target_node",
                }
            )
            continue

        merged_ids = [concept_id for concept_id in proposal.concept_ids if concept_id in node_map and concept_id != target_id]
        if not merged_ids:
            continue

        target_node = node_map[target_id]
        target_node.label = proposal.canonical_label or target_node.label
        target_node.aliases = sorted(
            dict.fromkeys(
                target_node.aliases
                + [node_map[concept_id].label for concept_id in merged_ids]
                + [
                    alias
                    for concept_id in merged_ids
                    for alias in node_map[concept_id].aliases
                ]
            )
        )
        target_node.metadata.setdefault("merged_from", []).extend(merged_ids)
        target_node.metadata.setdefault("merge_reasons", []).append(
            {
                "confidence_score": proposal.confidence_score,
                "reason": proposal.reason,
                "source": review.method,
            }
        )

        for merged_id in merged_ids:
            id_redirects[merged_id] = target_id
            node_map.pop(merged_id, None)

        report.applied_merges.append(
            {
                "target_id": target_id,
                "merged_ids": merged_ids,
                "canonical_label": target_node.label,
                "confidence_score": proposal.confidence_score,
                "reason": proposal.reason,
            }
        )

    rewritten_edges: list[GraphEdge] = []
    seen: set[tuple[str, str, str, str | None]] = set()
    for edge in edges:
        source = id_redirects.get(edge.source, edge.source)
        target = id_redirects.get(edge.target, edge.target)
        if source == target and edge.relation.value in {"related_to", "depends_on", "part_of", "causes", "semantically_similar_to"}:
            continue
        edge.source = source
        edge.target = target
        key = (edge.source, edge.target, edge.relation.value, edge.source_chunk_id)
        if key in seen:
            continue
        seen.add(key)
        rewritten_edges.append(edge)

    merged_graph = graph.model_copy(deep=True)
    merged_graph.nodes = list(node_map.values())
    merged_graph.edges = rewritten_edges
    merged_graph.metadata = dict(merged_graph.metadata)
    merged_graph.metadata["merge_application"] = report.to_dict()
    return merged_graph, report


def _should_apply(
    proposal: MergeProposal,
    auto_merge_threshold: float,
    assisted_merge_threshold: float,
) -> tuple[bool, str]:
    if proposal.confidence_score >= auto_merge_threshold:
        return True, "high_confidence"
    if proposal.confidence_score >= assisted_merge_threshold:
        return True, "assisted_confidence"
    return False, "below_threshold"


def _choose_target_id(proposal: MergeProposal, node_map: dict[str, GraphNode]) -> str | None:
    target_id: str | None = None
    longest = -1
    for concept_id in proposal.concept_ids:
        node = node_map.get(concept_id)
        if node is None or node.kind != NodeKind.concept:
            continue
        if node.label == proposal.canonical_label:
            return concept_id
        if len(node.label) > longest:
            longest = len(node.label)
            target_id = concept_id
    return target_id
