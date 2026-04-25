from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

from networkx.readwrite import json_graph

from src.document_understanding.chunking import (
    MarkdownBlock,
    MarkdownBlockKind,
    SemanticChunk,
    build_semantic_chunks,
    parse_markdown_blocks,
)
from src.document_understanding.extractor import DocumentExtractor
from src.document_understanding.parser import parse_document
from src.knowledge.concept_normalizer import ConceptMention, normalize_concept_mentions
from src.knowledge.graph_reviewer import GraphReviewReport, review_graph_for_merges
from src.knowledge.merge_resolver import MergeApplicationReport, apply_merge_proposals
from src.knowledge.schema import (
    ContentType,
    EdgeRelation,
    GraphEdge,
    GraphNode,
    MultimodalDocumentGraph,
    NodeKind,
    make_document_id,
)
from src.knowledge.schema_consolidator import consolidate_graph_schema
from src.knowledge.topic_inducer import TopicInductionResult, induce_topics
from src.knowledge.validator import GraphValidationReport, validate_graph
from src.utils.io import write_json


ATOMIC_ARTIFACT_KINDS = {
    MarkdownBlockKind.table,
    MarkdownBlockKind.code,
    MarkdownBlockKind.image,
    MarkdownBlockKind.details,
    MarkdownBlockKind.raw_html,
}


@dataclass
class KnowledgeGraphBuildResult:
    graph: MultimodalDocumentGraph
    checkpoints: dict[str, Any]
    validation: GraphValidationReport
    review: GraphReviewReport
    merge_application: MergeApplicationReport
    topics: TopicInductionResult | None = None


def build_knowledge_graph(
    document_understanding: dict[str, object],
    document_knowledge: dict[str, object] | None = None,
    *,
    source_file: str | Path | None = None,
    max_tokens: int = 280,
    overlap_blocks: int = 1,
    return_details: bool = False,
) -> MultimodalDocumentGraph | KnowledgeGraphBuildResult:
    result = build_knowledge_graph_workflow(
        document_understanding,
        document_knowledge=document_knowledge,
        source_file=source_file,
        max_tokens=max_tokens,
        overlap_blocks=overlap_blocks,
    )
    if return_details:
        return result
    return result.graph


def build_knowledge_graph_workflow(
    document_understanding: dict[str, object],
    document_knowledge: dict[str, object] | None = None,
    *,
    source_file: str | Path | None = None,
    max_tokens: int = 280,
    overlap_blocks: int = 1,
) -> KnowledgeGraphBuildResult:
    knowledge = document_knowledge or document_understanding or {}
    source_file_str = str(source_file) if source_file is not None else None
    document_id = make_document_id(source_file_str or "document")

    markdown_text = str(document_understanding.get("markdown", ""))
    sections = list(document_understanding.get("sections", []))
    paragraphs = list(document_understanding.get("paragraphs", []))
    figures = list(document_understanding.get("figures", []))
    captions = list(document_understanding.get("captions", []))

    if markdown_text.strip():
        blocks = parse_markdown_blocks(markdown_text, source_file=source_file_str)
    else:
        blocks = _fallback_blocks_from_legacy_fields(document_id, sections, paragraphs, figures, captions)

    chunks = build_semantic_chunks(blocks, max_tokens=max_tokens, overlap_blocks=overlap_blocks)
    block_to_chunk = _build_block_to_chunk_index(chunks)
    block_lookup = {block.id: block for block in blocks}
    chunk_lookup = {chunk.id: chunk for chunk in chunks}

    chunk_extractions = _normalize_chunk_extractions(knowledge.get("chunk_extractions", []), chunks, source_file_str)

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    checkpoints: dict[str, Any] = {}

    document_node = GraphNode(
        id=f"{document_id}::document",
        label=document_id,
        kind=NodeKind.document,
        content_type=ContentType.text,
        source_file=source_file_str,
        metadata={"source_document": source_file_str},
    )
    nodes.append(document_node)

    hierarchy_nodes, hierarchy_edges, section_lookup = _extract_hierarchy(document_id, blocks, source_file_str)
    nodes.extend(hierarchy_nodes)
    edges.extend(
        [
            GraphEdge(
                source=document_node.id,
                target=section_info["id"],
                relation=EdgeRelation.contains,
                confidence="EXTRACTED",
                confidence_score=1.0,
                source_file=source_file_str,
                extraction_method="deterministic",
            )
            for section_info in section_lookup.values()
            if section_info["parent_id"] is None
        ]
    )
    edges.extend(hierarchy_edges)
    checkpoints["hierarchy"] = {
        "section_count": len(hierarchy_nodes),
        "sections": [node.model_dump(mode="json") for node in hierarchy_nodes],
    }

    chunk_nodes, chunk_edges = _materialize_chunks(document_id, chunks, section_lookup, source_file_str)
    nodes.extend(chunk_nodes)
    edges.extend(chunk_edges)
    checkpoints["chunks"] = {
        "chunk_count": len(chunk_nodes),
        "chunks": [node.model_dump(mode="json") for node in chunk_nodes],
    }

    artifact_nodes, artifact_edges = _attach_artifacts(document_id, blocks, block_to_chunk, chunk_lookup, source_file_str)
    nodes.extend(artifact_nodes)
    edges.extend(artifact_edges)
    checkpoints["artifact_links"] = {
        "artifact_count": len(artifact_nodes),
        "artifacts": [node.model_dump(mode="json") for node in artifact_nodes],
    }

    concept_mentions = _collect_concept_mentions(chunk_extractions)
    normalization = normalize_concept_mentions(concept_mentions, document_id=document_id)
    checkpoints["canonicalization"] = {
        "canonical_concepts": [asdict(item) for item in normalization.canonical_concepts],
        "candidate_merges": [asdict(item) for item in normalization.candidate_merges],
        "summary": normalization.summary,
    }

    concept_nodes = _build_concept_nodes(normalization, source_file_str)
    nodes.extend(concept_nodes)

    semantic_edges = _build_semantic_edges(
        chunk_extractions,
        normalization.mention_to_concept,
        source_file=source_file_str,
    )
    artifact_semantic_edges = _link_artifacts_to_concepts(artifact_nodes, chunk_extractions, normalization.mention_to_concept)
    edges.extend(semantic_edges)
    edges.extend(artifact_semantic_edges)
    checkpoints["extraction_raw"] = {
        "chunk_extractions": chunk_extractions,
        "summary": knowledge.get("summary", {}),
    }

    base_graph = MultimodalDocumentGraph(
        document_id=document_id,
        source_file=source_file_str,
        nodes=_dedupe_nodes(nodes),
        edges=_dedupe_edges(edges),
        metadata={
            "sections": len(hierarchy_nodes),
            "chunks": len(chunk_nodes),
            "artifacts": len(artifact_nodes),
            "concepts": len(concept_nodes),
            "max_tokens": max_tokens,
            "overlap_blocks": overlap_blocks,
        },
    )

    review = review_graph_for_merges(normalization.canonical_concepts, normalization.candidate_merges)
    checkpoints["merge_review"] = review.to_dict()

    merged_graph, merge_application = apply_merge_proposals(base_graph, review)
    checkpoints["merge_application"] = merge_application.to_dict()

    consolidated_graph = consolidate_graph_schema(merged_graph)
    checkpoints["graph_consolidated"] = consolidated_graph.model_dump(mode="json")

    topic_result = induce_topics(consolidated_graph)
    checkpoints.update(topic_result.checkpoints())

    with_topics_graph = consolidated_graph.model_copy(deep=True)
    with_topics_graph.nodes = _dedupe_nodes(with_topics_graph.nodes + topic_result.topic_nodes)
    with_topics_graph.edges = _dedupe_edges(with_topics_graph.edges + topic_result.topic_edges)
    with_topics_graph.metadata = dict(with_topics_graph.metadata)
    with_topics_graph.metadata["topics"] = topic_result.summary

    validation = validate_graph(with_topics_graph)
    checkpoints["graph_validation"] = validation.to_dict()
    with_topics_graph.metadata = dict(with_topics_graph.metadata)
    with_topics_graph.metadata["validation"] = validation.to_dict()

    return KnowledgeGraphBuildResult(
        graph=with_topics_graph,
        checkpoints=checkpoints,
        validation=validation,
        review=review,
        merge_application=merge_application,
        topics=topic_result,
    )


def export_graph_bundle(
    graph: MultimodalDocumentGraph,
    *,
    output_dir: str | Path,
    html: bool = False,
    checkpoints: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_json_path = output_dir / f"{graph.document_id}_graph.json"
    write_json(graph_json_path, graph.model_dump(mode="json"))

    networkx_json_path = output_dir / f"{graph.document_id}_graph_networkx.json"
    graph_data = json_graph.node_link_data(graph.to_networkx())
    networkx_json_path.write_text(json.dumps(graph_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    artifacts: dict[str, Path] = {
        "graph_json": graph_json_path,
        "networkx_json": networkx_json_path,
    }

    if html:
        html_path = output_dir / f"{graph.document_id}_graph.html"
        _write_graph_html(graph, html_path)
        artifacts["html"] = html_path

    if checkpoints:
        checkpoint_map = {
            "hierarchy": "hierarchy.json",
            "chunks": "chunks.json",
            "artifact_links": "artifact_links.json",
            "extraction_raw": "extraction_raw.json",
            "canonicalization": "canonicalization.json",
            "merge_review": "merge_review.json",
            "merge_application": "merge_application.json",
            "graph_consolidated": "graph_consolidated.json",
            "topic_candidates": "topic_candidates.json",
            "topic_consolidation": "topic_consolidation.json",
            "topics": "topics.json",
            "graph_validation": "graph_validation.json",
        }
        for key, filename in checkpoint_map.items():
            if key not in checkpoints:
                continue
            path = output_dir / filename
            write_json(path, checkpoints[key])
            artifacts[key] = path

    return artifacts


def _extract_hierarchy(
    document_id: str,
    blocks: list[MarkdownBlock],
    source_file: str | None,
) -> tuple[list[GraphNode], list[GraphEdge], dict[str, dict[str, Any]]]:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    section_lookup: dict[str, dict[str, Any]] = {}
    section_stack: list[tuple[str, int, list[str]]] = []

    for index, block in enumerate(blocks):
        if block.kind != MarkdownBlockKind.heading:
            continue

        level = block.level or 1
        while section_stack and section_stack[-1][1] >= level:
            section_stack.pop()

        node_id = _safe_id(document_id, "section", index, block.text)
        section_path = list(block.section_path) + [block.text]
        nodes.append(
            GraphNode(
                id=node_id,
                label=block.text,
                kind=NodeKind.section,
                content_type=ContentType.text,
                source_file=source_file,
                section_path=section_path,
                metadata={"order": index, "level": level},
            )
        )

        parent_id = section_stack[-1][0] if section_stack else None
        if parent_id is not None:
            edges.append(
                GraphEdge(
                    source=parent_id,
                    target=node_id,
                    relation=EdgeRelation.contains,
                    confidence="EXTRACTED",
                    confidence_score=1.0,
                    source_file=source_file,
                    extraction_method="deterministic",
                )
            )

        section_lookup[" / ".join(section_path)] = {"id": node_id, "parent_id": parent_id, "section_path": section_path}
        section_stack.append((node_id, level, section_path))

    if not nodes:
        root_id = f"{document_id}::section::root"
        nodes.append(
            GraphNode(
                id=root_id,
                label="Document",
                kind=NodeKind.section,
                content_type=ContentType.text,
                source_file=source_file,
                section_path=["Document"],
                metadata={"order": 0, "level": 1, "synthetic": True},
            )
        )
        section_lookup["Document"] = {"id": root_id, "parent_id": None, "section_path": ["Document"]}

    return nodes, edges, section_lookup


def _materialize_chunks(
    document_id: str,
    chunks: list[SemanticChunk],
    section_lookup: dict[str, dict[str, Any]],
    source_file: str | None,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    previous_chunk_id: str | None = None

    for index, chunk in enumerate(chunks):
        node_id = chunk.id or _safe_id(document_id, "chunk", index, chunk.text)
        content_type = _chunk_content_type(chunk)
        nodes.append(
            GraphNode(
                id=node_id,
                label=_short_label(chunk.text),
                kind=NodeKind.chunk,
                content_type=content_type,
                source_file=source_file,
                source_chunk_id=node_id,
                section_path=list(chunk.section_path),
                text=chunk.text,
                metadata={
                    "block_ids": chunk.block_ids,
                    "block_kinds": [kind.value for kind in chunk.block_kinds],
                    "token_count": chunk.token_count,
                    **chunk.metadata,
                },
            )
        )

        section_key = " / ".join(chunk.section_path) if chunk.section_path else "Document"
        section_info = section_lookup.get(section_key)
        if section_info is None and section_lookup:
            section_info = next(iter(section_lookup.values()))
        if section_info is not None:
            edges.append(
                GraphEdge(
                    source=section_info["id"],
                    target=node_id,
                    relation=EdgeRelation.contains,
                    confidence="EXTRACTED",
                    confidence_score=1.0,
                    source_file=source_file,
                    source_chunk_id=node_id,
                    extraction_method="deterministic",
                )
            )

        if previous_chunk_id is not None:
            edges.append(
                GraphEdge(
                    source=previous_chunk_id,
                    target=node_id,
                    relation=EdgeRelation.follows,
                    confidence="EXTRACTED",
                    confidence_score=1.0,
                    source_file=source_file,
                    source_chunk_id=node_id,
                    extraction_method="deterministic",
                )
            )
        previous_chunk_id = node_id

    return nodes, edges


def _attach_artifacts(
    document_id: str,
    blocks: list[MarkdownBlock],
    block_to_chunk: dict[str, str],
    chunk_lookup: dict[str, SemanticChunk],
    source_file: str | None,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    for index, block in enumerate(blocks):
        if block.kind not in ATOMIC_ARTIFACT_KINDS:
            continue
        chunk_id = block_to_chunk.get(block.id)
        if chunk_id is None:
            continue
        artifact_id = _safe_id(document_id, "artifact", index, block.caption or block.text)
        nodes.append(
            GraphNode(
                id=artifact_id,
                label=_short_label(block.caption or block.text),
                kind=NodeKind.artifact,
                content_type=_block_content_type(block.kind),
                source_file=source_file,
                source_chunk_id=chunk_id,
                section_path=list(block.section_path),
                text=block.text,
                image_path=block.image_path,
                metadata={
                    "block_id": block.id,
                    "block_kind": block.kind.value,
                    **block.metadata,
                },
            )
        )
        edges.append(
            GraphEdge(
                source=chunk_id,
                target=artifact_id,
                relation=EdgeRelation.contains,
                confidence="EXTRACTED",
                confidence_score=1.0,
                source_file=source_file,
                source_chunk_id=chunk_id,
                extraction_method="deterministic",
            )
        )

    return nodes, edges


def _collect_concept_mentions(chunk_extractions: list[dict[str, Any]]) -> list[ConceptMention]:
    mentions: list[ConceptMention] = []
    for row in chunk_extractions:
        chunk_id = _as_optional_str(row.get("chunk_id"))
        source_file = _as_optional_str(row.get("source_file"))
        section_path = [str(item) for item in row.get("section_path", [])]
        definitions = {
            str(item.get("concept", "")).strip(): str(item.get("definition", "")).strip()
            for item in row.get("definitions", [])
            if isinstance(item, dict)
        }
        for concept in row.get("concepts", []):
            label = str(concept).strip()
            if not label:
                continue
            mentions.append(
                ConceptMention(
                    label=label,
                    chunk_id=chunk_id,
                    source_file=source_file,
                    section_path=section_path,
                    definition=definitions.get(label),
                    metadata={"extraction_method": row.get("extraction_method", "unknown")},
                )
            )
        for concept, definition in definitions.items():
            if concept and definition:
                mentions.append(
                    ConceptMention(
                        label=concept,
                        chunk_id=chunk_id,
                        source_file=source_file,
                        section_path=section_path,
                        definition=definition,
                        metadata={"extraction_method": row.get("extraction_method", "unknown"), "from_definition": True},
                    )
                )
    return mentions


def _build_concept_nodes(normalization, source_file: str | None) -> list[GraphNode]:
    nodes: list[GraphNode] = []
    for concept in normalization.canonical_concepts:
        nodes.append(
            GraphNode(
                id=concept.id,
                label=concept.label,
                kind=NodeKind.concept,
                content_type=ContentType.text,
                source_file=source_file,
                section_path=concept.section_paths[0] if concept.section_paths else [],
                aliases=list(concept.aliases),
                metadata={
                    "mention_count": concept.mention_count,
                    "definitions": concept.definitions,
                    "source_chunk_ids": concept.source_chunk_ids,
                },
            )
        )
    return nodes


def _build_semantic_edges(
    chunk_extractions: list[dict[str, Any]],
    mention_to_concept: dict[str, str],
    *,
    source_file: str | None,
) -> list[GraphEdge]:
    edges: list[GraphEdge] = []

    for row in chunk_extractions:
        chunk_id = _as_optional_str(row.get("chunk_id"))
        if not chunk_id:
            continue
        extraction_method = str(row.get("extraction_method", "unknown"))

        for concept in row.get("concepts", []):
            concept_name = str(concept).strip()
            concept_id = mention_to_concept.get(f"{chunk_id}::{_normalize_lookup_key(concept_name)}")
            if concept_id is None:
                continue
            edges.append(
                GraphEdge(
                    source=chunk_id,
                    target=concept_id,
                    relation=EdgeRelation.mentions,
                    confidence="EXTRACTED",
                    confidence_score=1.0,
                    source_file=source_file,
                    source_chunk_id=chunk_id,
                    extraction_method=extraction_method,
                )
            )

        for definition in row.get("definitions", []):
            if not isinstance(definition, dict):
                continue
            concept_name = str(definition.get("concept", "")).strip()
            definition_text = str(definition.get("definition", "")).strip()
            concept_id = mention_to_concept.get(f"{chunk_id}::{_normalize_lookup_key(concept_name)}")
            if concept_id is None:
                continue
            edges.append(
                GraphEdge(
                    source=chunk_id,
                    target=concept_id,
                    relation=EdgeRelation.defines,
                    confidence="EXTRACTED",
                    confidence_score=1.0,
                    source_file=source_file,
                    source_chunk_id=chunk_id,
                    extraction_method=extraction_method,
                    metadata={"definition": definition_text},
                )
            )

        for example in row.get("examples", []):
            example_text = str(example).strip()
            if not example_text:
                continue
            concept_id = _find_related_concept_for_text(example_text, mention_to_concept, chunk_id)
            if concept_id is None:
                continue
            edges.append(
                GraphEdge(
                    source=chunk_id,
                    target=concept_id,
                    relation=EdgeRelation.supports,
                    confidence="INFERRED",
                    confidence_score=0.7,
                    source_file=source_file,
                    source_chunk_id=chunk_id,
                    extraction_method=extraction_method,
                    metadata={"example": example_text},
                )
            )

        for relation in row.get("relations", []):
            if not isinstance(relation, dict):
                continue
            source_name = str(relation.get("source", "")).strip()
            target_name = str(relation.get("target", "")).strip()
            source_id = mention_to_concept.get(f"{chunk_id}::{_normalize_lookup_key(source_name)}")
            target_id = mention_to_concept.get(f"{chunk_id}::{_normalize_lookup_key(target_name)}")
            if source_id is None or target_id is None:
                source_id = source_id or _lookup_global_concept(mention_to_concept, source_name)
                target_id = target_id or _lookup_global_concept(mention_to_concept, target_name)
            if source_id is None or target_id is None:
                continue
            edges.append(
                GraphEdge(
                    source=source_id,
                    target=target_id,
                    relation=_normalize_relation(str(relation.get("relation", "related_to"))),
                    confidence=str(relation.get("confidence", "EXTRACTED")),
                    confidence_score=float(relation.get("confidence_score", 1.0)),
                    source_file=source_file,
                    source_chunk_id=chunk_id,
                    extraction_method=str(relation.get("extraction_method", extraction_method)),
                )
            )

    return edges


def _link_artifacts_to_concepts(
    artifact_nodes: list[GraphNode],
    chunk_extractions: list[dict[str, Any]],
    mention_to_concept: dict[str, str],
) -> list[GraphEdge]:
    chunk_to_concepts: dict[str, set[str]] = {}
    for row in chunk_extractions:
        chunk_id = _as_optional_str(row.get("chunk_id"))
        if not chunk_id:
            continue
        for concept in row.get("concepts", []):
            concept_id = mention_to_concept.get(f"{chunk_id}::{_normalize_lookup_key(str(concept))}")
            if concept_id:
                chunk_to_concepts.setdefault(chunk_id, set()).add(concept_id)

    edges: list[GraphEdge] = []
    for artifact in artifact_nodes:
        if not artifact.source_chunk_id:
            continue
        relation = EdgeRelation.illustrates if artifact.content_type == ContentType.image else EdgeRelation.supports
        for concept_id in sorted(chunk_to_concepts.get(artifact.source_chunk_id, set()))[:5]:
            edges.append(
                GraphEdge(
                    source=artifact.id,
                    target=concept_id,
                    relation=relation,
                    confidence="INFERRED",
                    confidence_score=0.65,
                    source_file=artifact.source_file,
                    source_chunk_id=artifact.source_chunk_id,
                    extraction_method="artifact_linker",
                )
            )
    return edges


def _normalize_chunk_extractions(
    rows: object,
    chunks: list[SemanticChunk],
    source_file: str | None,
) -> list[dict[str, Any]]:
    if isinstance(rows, list) and rows:
        normalized: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            chunk = chunks[index] if index < len(chunks) else None
            normalized.append(
                {
                    "chunk_index": int(row.get("chunk_index", index)),
                    "chunk_id": row.get("chunk_id") or (chunk.id if chunk else None),
                    "source_file": row.get("source_file") or source_file,
                    "section_path": list(row.get("section_path", chunk.section_path if chunk else [])),
                    "concepts": list(row.get("concepts", [])),
                    "definitions": list(row.get("definitions", [])),
                    "relations": list(row.get("relations", [])),
                    "examples": list(row.get("examples", [])),
                    "extraction_method": row.get("extraction_method", "unknown"),
                }
            )
        return normalized

    return [
        {
            "chunk_index": index,
            "chunk_id": chunk.id,
            "source_file": source_file,
            "section_path": list(chunk.section_path),
            "concepts": [],
            "definitions": [],
            "relations": [],
            "examples": [],
            "extraction_method": "none",
        }
        for index, chunk in enumerate(chunks)
    ]


def _safe_id(document_id: str, kind: str, index: object, text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")[:48]
    suffix = slug or "item"
    return f"{document_id}::{kind}::{index}::{suffix}"


def _short_label(text: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _chunk_content_type(chunk: SemanticChunk) -> ContentType:
    content_types = {_block_content_type(kind) for kind in chunk.block_kinds}
    if not content_types:
        return ContentType.text
    if len(content_types) == 1:
        return next(iter(content_types))
    return ContentType.mixed


def _block_content_type(kind: MarkdownBlockKind) -> ContentType:
    return {
        MarkdownBlockKind.paragraph: ContentType.text,
        MarkdownBlockKind.heading: ContentType.text,
        MarkdownBlockKind.table: ContentType.table,
        MarkdownBlockKind.code: ContentType.code,
        MarkdownBlockKind.image: ContentType.image,
        MarkdownBlockKind.details: ContentType.details,
        MarkdownBlockKind.list_item: ContentType.list,
        MarkdownBlockKind.quote: ContentType.text,
        MarkdownBlockKind.raw_html: ContentType.raw_html,
    }.get(kind, ContentType.text)


def _build_block_to_chunk_index(chunks: list[SemanticChunk]) -> dict[str, str]:
    block_to_chunk: dict[str, str] = {}
    for chunk in chunks:
        for block_id in chunk.block_ids:
            block_to_chunk[block_id] = chunk.id
    return block_to_chunk


def _normalize_relation(relation_name: str) -> EdgeRelation:
    candidate = relation_name.strip().casefold()
    relation_map = {
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
    }
    return relation_map.get(candidate, EdgeRelation.related_to)


def _normalize_lookup_key(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[_/]", " ", text)
    text = re.sub(r"[^a-z0-9+\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" -")


def _find_related_concept_for_text(text: str, mention_to_concept: dict[str, str], chunk_id: str) -> str | None:
    normalized_text = _normalize_lookup_key(text)
    for key, concept_id in mention_to_concept.items():
        key_chunk_id, _, concept_name = key.partition("::")
        if key_chunk_id != chunk_id:
            continue
        if concept_name and concept_name in normalized_text:
            return concept_id
    return None


def _lookup_global_concept(mention_to_concept: dict[str, str], name: str) -> str | None:
    normalized = _normalize_lookup_key(name)
    for key, concept_id in mention_to_concept.items():
        if key.endswith(f"::{normalized}"):
            return concept_id
    return None


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _fallback_blocks_from_legacy_fields(
    document_id: str,
    sections: list[str],
    paragraphs: list[str],
    figures: list[str],
    captions: list[str],
) -> list[MarkdownBlock]:
    blocks: list[MarkdownBlock] = []
    current_path: list[str] = []

    for index, section in enumerate(sections):
        current_path = [section]
        blocks.append(
            MarkdownBlock(
                id=_safe_id(document_id, "heading", index, section),
                kind=MarkdownBlockKind.heading,
                text=section,
                section_path=list(current_path[:-1]),
                level=1,
                metadata={"legacy": True},
            )
        )

    for index, paragraph in enumerate(paragraphs):
        blocks.append(
            MarkdownBlock(
                id=_safe_id(document_id, "paragraph", index, paragraph),
                kind=MarkdownBlockKind.paragraph,
                text=paragraph,
                section_path=list(current_path),
                metadata={"legacy": True},
            )
        )

    for index, figure in enumerate(figures):
        caption = captions[index] if index < len(captions) else None
        blocks.append(
            MarkdownBlock(
                id=_safe_id(document_id, "image", index, caption or figure),
                kind=MarkdownBlockKind.image,
                text=caption or figure,
                section_path=list(current_path),
                image_path=figure,
                caption=caption,
                metadata={"legacy": True},
            )
        )

    return blocks


def _dedupe_nodes(nodes: Iterable[GraphNode]) -> list[GraphNode]:
    seen: set[str] = set()
    deduped: list[GraphNode] = []
    for node in nodes:
        if node.id in seen:
            continue
        seen.add(node.id)
        deduped.append(node)
    return deduped


def _dedupe_edges(edges: Iterable[GraphEdge]) -> list[GraphEdge]:
    seen: set[tuple[str, str, str, str | None]] = set()
    deduped: list[GraphEdge] = []
    for edge in edges:
        key = (edge.source, edge.target, edge.relation.value, edge.source_chunk_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def _write_graph_html(graph: MultimodalDocumentGraph, output_path: Path) -> None:
    try:
        from pyvis.network import Network  # type: ignore[import-not-found]
    except Exception:
        output_path.write_text(_fallback_graph_html(graph), encoding="utf-8")
        return

    network = Network(height="900px", width="100%", directed=True, bgcolor="#ffffff", font_color="#111111")
    network.barnes_hut()

    for node in graph.nodes:
        title_parts = [f"<b>{node.label}</b>", f"kind: {node.kind.value}"]
        if node.content_type:
            title_parts.append(f"content_type: {node.content_type.value}")
        if node.section_path:
            title_parts.append(f"section: {' / '.join(node.section_path)}")
        if node.aliases:
            title_parts.append(f"aliases: {', '.join(node.aliases[:8])}")
        if node.text:
            title_parts.append(f"text: {node.text[:500]}")
        title = "<br>".join(title_parts)
        network.add_node(node.id, label=node.label, title=title, color=_node_color(node.kind), shape="dot")

    for edge in graph.edges:
        network.add_edge(
            edge.source,
            edge.target,
            label=edge.relation.value,
            title=f"{edge.relation.value} ({edge.confidence}, {edge.confidence_score:.2f})",
            value=edge.weight,
        )

    network.write_html(str(output_path))


def _fallback_graph_html(graph: MultimodalDocumentGraph) -> str:
    summary = graph.summary()
    node_rows = "\n".join(f"<li><b>{node.label}</b> [{node.kind.value}] - {node.id}</li>" for node in graph.nodes[:200])
    edge_rows = "\n".join(
        f"<li>{edge.source} -> {edge.target} ({edge.relation.value}, {edge.confidence}, {edge.confidence_score:.2f})</li>"
        for edge in graph.edges[:200]
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{graph.document_id} graph</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    ul {{ line-height: 1.5; }}
  </style>
</head>
<body>
  <h1>{graph.document_id}</h1>
  <p><b>Nodes:</b> {summary['node_count']} | <b>Edges:</b> {summary['edge_count']}</p>
  <div class="grid">
    <section>
      <h2>Nodes</h2>
      <ul>{node_rows}</ul>
    </section>
    <section>
      <h2>Edges</h2>
      <ul>{edge_rows}</ul>
    </section>
  </div>
</body>
</html>"""


def _node_color(kind: NodeKind) -> str:
    return {
        NodeKind.document: "#2E4057",
        NodeKind.section: "#4C78A8",
        NodeKind.chunk: "#72B7B2",
        NodeKind.concept: "#F58518",
        NodeKind.topic: "#54A24B",
        NodeKind.artifact: "#E45756",
        NodeKind.image: "#E45756",
        NodeKind.block: "#72B7B2",
    }.get(kind, "#999999")


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    if load_dotenv is not None:
        load_dotenv(project_root / ".env")

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    logger = logging.getLogger("kg_builder")

    parser = argparse.ArgumentParser(description="Build and visualize a multimodal document graph from a markdown file.")
    parser.add_argument("markdown_path", type=Path, help="Path to the markdown file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for graph artifacts")
    parser.add_argument("--html", action="store_true", help="Generate HTML visualization")
    parser.add_argument(
        "--backend",
        choices=["rule", "langchain"],
        default=os.getenv("QUIZGEN_EXTRACTOR_BACKEND", "langchain"),
        help="Extraction backend",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "mistral"],
        default=os.getenv("QUIZGEN_LLM_PROVIDER", "openai"),
        help="LLM provider",
    )
    parser.add_argument(
        "--granularity",
        choices=["coarse", "balanced", "fine"],
        default=os.getenv("QUIZGEN_EXTRACTION_GRANULARITY", "balanced"),
        help="Extraction granularity",
    )
    parser.add_argument("--max-tokens", type=int, default=280, help="Semantic chunk token budget")
    parser.add_argument("--overlap-blocks", type=int, default=1, help="Block overlap between chunks")
    args = parser.parse_args()

    markdown_path = args.markdown_path
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    logger.info("Step 1: Parsing markdown file...")
    parsed_document = parse_document(markdown_path)

    blocks = parse_markdown_blocks(parsed_document.markdown, source_file=markdown_path)
    chunks = build_semantic_chunks(blocks, max_tokens=args.max_tokens, overlap_blocks=args.overlap_blocks)
    logger.info("Step 2: Extracting semantic knowledge from %d chunks...", len(chunks))
    extractor = DocumentExtractor(
        backend=args.backend,
        provider=args.provider,
        granularity=args.granularity,
        batch_size=8,
        max_calls=24,
    )
    extracted = extractor.extract_chunks(chunks, source_file=str(markdown_path))

    logger.info("Step 3: Building knowledge graph...")
    result = build_knowledge_graph_workflow(
        {
            "markdown": parsed_document.markdown,
            "sections": parsed_document.sections,
            "paragraphs": parsed_document.paragraphs,
            "figures": parsed_document.figures,
            "captions": parsed_document.captions,
        },
        extracted,
        source_file=markdown_path,
        max_tokens=args.max_tokens,
        overlap_blocks=args.overlap_blocks,
    )
    logger.info("  Built graph: %s", result.graph.summary())
    if not result.validation.passed:
        logger.warning("Graph validation failed: %s", "; ".join(result.validation.errors))

    logger.info("Step 4: Exporting artifacts...")
    output_dir = args.output_dir or markdown_path.parent
    artifacts = export_graph_bundle(result.graph, output_dir=output_dir, html=args.html, checkpoints=result.checkpoints)
    logger.info("Saved graph JSON to %s", artifacts["graph_json"])
    logger.info("Saved NetworkX JSON to %s", artifacts["networkx_json"])
    if "graph_validation" in artifacts:
        logger.info("Saved validation report to %s", artifacts["graph_validation"])
    print(f"\nPipeline complete. Graph summary: {result.graph.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
