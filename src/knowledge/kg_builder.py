from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable
import os
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None  # type: ignore[assignment]

from src.document_understanding.parser import ParsedDocument, parse_document

from src.document_understanding.extractor import DocumentExtractor

from networkx.readwrite import json_graph

from src.document_understanding.chunking import (
    MarkdownBlock,
    MarkdownBlockKind,
    SemanticChunk,
    build_semantic_chunks,
    parse_markdown_blocks,
)
from src.knowledge.schema import (
    EdgeRelation,
    GraphEdge,
    GraphNode,
    MultimodalDocumentGraph,
    NodeKind,
    make_document_id,
)
from src.utils.io import write_json


def build_knowledge_graph(
    document_understanding: dict[str, object],
    document_knowledge: dict[str, object] | None = None,
    *,
    source_file: str | Path | None = None,
    max_tokens: int = 280,
    overlap_blocks: int = 1,
) -> MultimodalDocumentGraph:
    """Build a multimodal document graph from parsed markdown and extracted knowledge."""
    knowledge = document_knowledge or document_understanding or {}
    source_file_str = str(source_file) if source_file is not None else None
    document_id = make_document_id(source_file_str or "document")

    markdown_text = str(document_understanding.get("markdown", ""))
    sections = list(document_understanding.get("sections", []))
    paragraphs = list(document_understanding.get("paragraphs", []))
    figures = list(document_understanding.get("figures", []))
    captions = list(document_understanding.get("captions", []))

    blocks: list[MarkdownBlock]
    if markdown_text.strip():
        blocks = parse_markdown_blocks(markdown_text, source_file=source_file_str)
    else:
        blocks = _fallback_blocks_from_legacy_fields(document_id, sections, paragraphs, figures, captions)

    chunks = build_semantic_chunks(blocks, max_tokens=max_tokens, overlap_blocks=overlap_blocks)
    block_to_chunk = _build_block_to_chunk_index(chunks)

    concepts = list(knowledge.get("concepts", []))
    definitions = dict(knowledge.get("definitions", {}))
    relations = list(knowledge.get("relations", []))
    examples = list(knowledge.get("examples", []))

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    section_stack: list[tuple[str, str, int]] = []
    previous_content_node_id: str | None = None
    current_section_node_id: str | None = None

    for index, block in enumerate(blocks):
        if block.kind == MarkdownBlockKind.heading:
            node_id = _safe_id(document_id, "section", index, block.text)
            level = block.level or _section_level(block.text)
            section_stack[:] = [item for item in section_stack if item[2] < level]

            nodes.append(
                GraphNode(
                    id=node_id,
                    label=block.text,
                    kind=NodeKind.section,
                    source_file=source_file_str,
                    section_path=block.section_path,
                    metadata={"order": index, "level": level, "block_kind": block.kind.value},
                )
            )

            if section_stack:
                edges.append(
                    GraphEdge(
                        source=section_stack[-1][0],
                        target=node_id,
                        relation=EdgeRelation.contains,
                        confidence="EXTRACTED",
                        confidence_score=1.0,
                        source_file=source_file_str,
                    )
                )

            section_stack.append((node_id, block.text, level))
            current_section_node_id = node_id
            previous_content_node_id = None
            continue

        node_kind = NodeKind.image if block.kind == MarkdownBlockKind.image else NodeKind.block
        node_id = _safe_id(document_id, node_kind.value, index, block.text)
        current_chunk_id = block_to_chunk.get(block.id)

        nodes.append(
            GraphNode(
                id=node_id,
                label=_short_label(block.caption or block.text),
                kind=node_kind,
                source_file=source_file_str,
                section_path=block.section_path,
                text=block.text,
                image_path=block.image_path,
                metadata={
                    "order": index,
                    "block_kind": block.kind.value,
                    "chunk_id": current_chunk_id,
                    **block.metadata,
                },
            )
        )

        if current_section_node_id is not None:
            edges.append(
                GraphEdge(
                    source=current_section_node_id,
                    target=node_id,
                    relation=EdgeRelation.contains,
                    confidence="EXTRACTED",
                    confidence_score=1.0,
                    source_file=source_file_str,
                )
            )

        if previous_content_node_id is not None and block.kind not in ATOMIC_FOLLOW_BREAK_KINDS:
            edges.append(
                GraphEdge(
                    source=previous_content_node_id,
                    target=node_id,
                    relation=EdgeRelation.follows,
                    confidence="EXTRACTED",
                    confidence_score=1.0,
                    source_file=source_file_str,
                )
            )

        if node_kind == NodeKind.image:
            nearest_block_id = _nearest_block_node_id(nodes)
            if nearest_block_id is not None and nearest_block_id != node_id:
                edges.append(
                    GraphEdge(
                        source=nearest_block_id,
                        target=node_id,
                        relation=EdgeRelation.illustrates,
                        confidence="INFERRED",
                        confidence_score=0.75,
                        source_file=source_file_str,
                        metadata={"reason": "image_near_text"},
                    )
                )

        previous_content_node_id = node_id

    concept_nodes: dict[str, str] = {}
    for index, concept in enumerate(concepts):
        node_id = _safe_id(document_id, "concept", index, concept)
        concept_nodes[concept.lower()] = node_id
        nodes.append(
            GraphNode(
                id=node_id,
                label=concept,
                kind=NodeKind.concept,
                source_file=source_file_str,
                metadata={"order": index, "concept_type": "extracted"},
            )
        )

    for concept_name, definition in definitions.items():
        concept_node_id = concept_nodes.get(concept_name.lower())
        if concept_node_id is None:
            continue
        definition_block_id = _safe_id(document_id, "definition", concept_name, definition)
        nodes.append(
            GraphNode(
                id=definition_block_id,
                label=_short_label(definition),
                kind=NodeKind.block,
                source_file=source_file_str,
                text=definition,
                metadata={"block_type": "definition"},
            )
        )
        edges.append(
            GraphEdge(
                source=definition_block_id,
                target=concept_node_id,
                relation=EdgeRelation.defines,
                confidence="EXTRACTED",
                confidence_score=1.0,
                source_file=source_file_str,
            )
        )

    for relation in relations:
        source_name = str(relation.get("source", "")).strip()
        target_name = str(relation.get("target", "")).strip()
        relation_name = str(relation.get("relation", "related_to")).strip()
        if not source_name or not target_name:
            continue

        source_id = _lookup_concept_node_id(concept_nodes, source_name)
        target_id = _lookup_concept_node_id(concept_nodes, target_name)
        if source_id is None or target_id is None:
            continue

        edges.append(
            GraphEdge(
                source=source_id,
                target=target_id,
                relation=_normalize_relation(relation_name),
                confidence="EXTRACTED",
                confidence_score=1.0,
                source_file=source_file_str,
            )
        )

    for index, example in enumerate(examples):
        example_node_id = _safe_id(document_id, "example", index, example)
        nodes.append(
            GraphNode(
                id=example_node_id,
                label=_short_label(example),
                kind=NodeKind.block,
                source_file=source_file_str,
                text=example,
                metadata={"block_type": "example"},
            )
        )

        related_concept_id = _find_related_concept_node(example, concept_nodes)
        if related_concept_id is not None:
            edges.append(
                GraphEdge(
                    source=example_node_id,
                    target=related_concept_id,
                    relation=EdgeRelation.explains,
                    confidence="INFERRED",
                    confidence_score=0.7,
                    source_file=source_file_str,
                )
            )

    return MultimodalDocumentGraph(
        document_id=document_id,
        source_file=source_file_str,
        nodes=_dedupe_nodes(nodes),
        edges=_dedupe_edges(edges),
        metadata={
            "sections": len(sections),
            "paragraphs": len(paragraphs),
            "figures": len(figures),
            "concepts": len(concepts),
            "chunks": len(chunks),
        },
    )


def _safe_id(document_id: str, kind: str, index: object, text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48]
    suffix = slug or "item"
    return f"{document_id}::{kind}::{index}::{suffix}"


def _short_label(text: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _section_level(section_title: str) -> int:
    match = re.match(r"^(\d+(?:\.\d+)*)", section_title)
    if not match:
        return 1
    return match.group(1).count(".") + 1


def _lookup_concept_node_id(concept_nodes: dict[str, str], name: str) -> str | None:
    return concept_nodes.get(name.lower())


def _normalize_relation(relation_name: str) -> EdgeRelation:
    candidate = relation_name.lower().strip()
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
        "semantically_similar_to": EdgeRelation.semantically_similar_to,
        "causes": EdgeRelation.related_to,
        "depends_on": EdgeRelation.related_to,
        "part_of": EdgeRelation.related_to,
    }
    return relation_map.get(candidate, EdgeRelation.related_to)


def _nearest_block_node_id(nodes: list[GraphNode]) -> str | None:
    for node in reversed(nodes):
        if node.kind == NodeKind.block:
            return node.id
    return None


ATOMIC_FOLLOW_BREAK_KINDS = {
    MarkdownBlockKind.table,
    MarkdownBlockKind.code,
    MarkdownBlockKind.image,
    MarkdownBlockKind.details,
    MarkdownBlockKind.raw_html,
}


def _build_block_to_chunk_index(chunks: list[SemanticChunk]) -> dict[str, str]:
    block_to_chunk: dict[str, str] = {}
    for chunk in chunks:
        for block_id in chunk.block_ids:
            block_to_chunk[block_id] = chunk.id
    return block_to_chunk


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
                level=_section_level(section),
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


def _find_related_concept_node(text: str, concept_nodes: dict[str, str]) -> str | None:
    lowered_text = text.lower()
    for concept_name, node_id in concept_nodes.items():
        if concept_name in lowered_text:
            return node_id
    return None


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
    seen: set[tuple[str, str, str]] = set()
    deduped: list[GraphEdge] = []
    for edge in edges:
        key = (edge.source, edge.target, edge.relation.value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def export_graph_bundle(
    graph: MultimodalDocumentGraph,
    *,
    output_dir: str | Path,
    html: bool = False,
) -> dict[str, Path]:
    """Write JSON and optional HTML visualizations for a document graph."""
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

    return artifacts


def _write_graph_html(graph: MultimodalDocumentGraph, output_path: Path) -> None:
    try:
        from pyvis.network import Network  # type: ignore[import-not-found]
    except Exception:
        output_path.write_text(_fallback_graph_html(graph), encoding="utf-8")
        return

    network = Network(height="900px", width="100%", directed=True, bgcolor="#ffffff", font_color="#111111")
    network.barnes_hut()

    for node in graph.nodes:
        color = _node_color(node.kind)
        title_parts = [f"<b>{node.label}</b>", f"kind: {node.kind.value}"]
        if node.section_path:
            title_parts.append(f"section: {' / '.join(node.section_path)}")
        if node.text:
            title_parts.append(f"text: {node.text[:500]}")
        if node.image_path:
            title_parts.append(f"image: {node.image_path}")
        title = "<br>".join(title_parts)
        network.add_node(node.id, label=node.label, title=title, color=color, shape="dot")

    for edge in graph.edges:
        network.add_edge(
            edge.source,
            edge.target,
            label=edge.relation.value,
            title=f"{edge.relation.value} ({edge.confidence}, {edge.confidence_score:.2f})",
            value=edge.weight,
        )

    network.set_options(
        """
        {
          "nodes": {
            "font": {"size": 18},
            "borderWidth": 1
          },
          "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
            "smooth": {"type": "dynamic"}
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 160,
              "springConstant": 0.04,
              "damping": 0.12
            }
          }
        }
        """
    )
    network.write_html(str(output_path))


def _fallback_graph_html(graph: MultimodalDocumentGraph) -> str:
    summary = graph.summary()
    node_rows = "\n".join(
        f"<li><b>{node.label}</b> [{node.kind.value}] - {node.id}</li>" for node in graph.nodes[:200]
    )
    edge_rows = "\n".join(
        f"<li>{edge.source} → {edge.target} ({edge.relation.value}, {edge.confidence}, {edge.confidence_score:.2f})</li>"
        for edge in graph.edges[:200]
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>{graph.document_id} graph</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    ul {{ line-height: 1.5; }}
    code {{ background: #f6f6f6; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>{graph.document_id}</h1>
  <p><b>Nodes:</b> {summary['node_count']} | <b>Edges:</b> {summary['edge_count']}</p>
  <div class=\"grid\">
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
        NodeKind.section: "#4C78A8",
        NodeKind.block: "#72B7B2",
        NodeKind.concept: "#F58518",
        NodeKind.image: "#E45756",
    }.get(kind, "#999999")


def main() -> int:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    logger = logging.getLogger("kg_builder")

    parser = argparse.ArgumentParser(
        description="Build and visualize a multimodal document graph from a markdown file. "
        "Runs the full pipeline: parse → extract → build KG → visualize."
    )
    parser.add_argument("markdown_path", type=Path, help="Path to the markdown file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for graph artifacts (defaults to the markdown folder)",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML visualization (uses pyvis if installed)",
    )
    parser.add_argument(
        "--backend",
        choices=["rule", "langchain"],
        default=os.getenv("QUIZGEN_EXTRACTOR_BACKEND", "langchain"),
        help="Extraction backend (default from env)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "mistral"],
        default=os.getenv("QUIZGEN_LLM_PROVIDER", "openai"),
        help="LLM provider (default from env)",
    )
    parser.add_argument(
        "--granularity",
        choices=["coarse", "balanced", "fine"],
        default=os.getenv("QUIZGEN_EXTRACTION_GRANULARITY", "balanced"),
        help="Extraction granularity (default from env)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=280,
        help="Semantic chunk token budget",
    )
    parser.add_argument(
        "--overlap-blocks",
        type=int,
        default=1,
        help="Previous blocks to overlap between chunks",
    )
    args = parser.parse_args()

    markdown_path = args.markdown_path
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    logger.info("Step 1: Parsing markdown file...")
    parsed_document = parse_document(markdown_path)
    logger.info(
        "  Parsed: %d sections, %d paragraphs, %d figures",
        len(parsed_document.sections),
        len(parsed_document.paragraphs),
        len(parsed_document.figures),
    )

    logger.info("Step 2: Extracting semantic knowledge...")
    extractor = DocumentExtractor(
        backend=args.backend,
        provider=args.provider,
        granularity=args.granularity,
        batch_size=8,
        max_calls=24,
    )
    extracted = extractor.extract(parsed_document.markdown) if parsed_document.markdown.strip() else {
        "concepts": [],
        "definitions": {},
        "relations": [],
        "examples": [],
    }
    logger.info(
        "  Extracted: %d concepts, %d definitions, %d relations, %d examples",
        len(extracted.get("concepts", [])),
        len(extracted.get("definitions", {})),
        len(extracted.get("relations", [])),
        len(extracted.get("examples", [])),
    )

    logger.info("Step 3: Building knowledge graph...")
    graph = build_knowledge_graph(
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
    logger.info("  Built graph: %s", graph.summary())

    logger.info("Step 4: Exporting artifacts...")
    output_dir = args.output_dir or markdown_path.parent
    artifacts = export_graph_bundle(graph, output_dir=output_dir, html=args.html)
    logger.info("Saved graph JSON to %s", artifacts["graph_json"])
    logger.info("Saved NetworkX JSON to %s", artifacts["networkx_json"])
    if "html" in artifacts:
        logger.info("Saved HTML visualization to %s", artifacts["html"])
    print(f"\n✓ Pipeline complete. Graph summary: {graph.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())