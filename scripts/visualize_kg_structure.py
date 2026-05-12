from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


NODE_COLORS: dict[str, str] = {
    "document": "#2E4057",
    "section": "#4C78A8",
    "chunk": "#72B7B2",
    "concept": "#F58518",
    "topic": "#54A24B",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def short_label(text: str, limit: int = 42) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def build_indexes(graph_payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    nodes = graph_payload.get("nodes", [])
    edges = graph_payload.get("edges", [])
    node_by_id = {node["id"]: node for node in nodes if "id" in node}
    return node_by_id, edges


def pick_document(node_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for node in node_by_id.values():
        if node.get("kind") == "document":
            return node
    raise ValueError("No document node found in graph JSON.")


def pick_sections(
    document_id: str,
    node_by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    max_sections: int = 4,
) -> list[dict[str, Any]]:
    section_ids: list[str] = []
    for edge in edges:
        if edge.get("relation") == "contains" and edge.get("source") == document_id:
            target = edge.get("target")
            if target in node_by_id and node_by_id[target].get("kind") == "section":
                section_ids.append(target)

    if not section_ids:
        section_ids = [node["id"] for node in node_by_id.values() if node.get("kind") == "section"]

    def section_sort_key(section_id: str) -> tuple[int, str]:
        metadata = node_by_id[section_id].get("metadata") or {}
        order_value = metadata.get("order")
        if isinstance(order_value, int):
            return (order_value, section_id)
        return (10**9, section_id)

    unique_ids = sorted(set(section_ids), key=section_sort_key)
    selected = [node_by_id[section_id] for section_id in unique_ids[:max_sections]]

    if len(selected) < max_sections:
        all_sections = [node for node in node_by_id.values() if node.get("kind") == "section"]
        seen = {node["id"] for node in selected}
        for section in sorted(all_sections, key=lambda n: n.get("label", "")):
            if section["id"] in seen:
                continue
            selected.append(section)
            if len(selected) == max_sections:
                break

    if len(selected) < 4:
        raise ValueError("Need at least 4 section nodes to build the demonstration figure.")
    return selected[:4]


def chunk_ids_for_section(
    section: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    per_section: int,
) -> list[str]:
    matched: list[str] = []
    section_id = section["id"]

    for edge in edges:
        if edge.get("relation") != "contains":
            continue
        if edge.get("source") != section_id:
            continue
        target = edge.get("target")
        if target in node_by_id and node_by_id[target].get("kind") == "chunk":
            matched.append(target)

    if len(matched) < per_section:
        section_path = section.get("section_path") or []
        section_title = section_path[-1] if section_path else section.get("label")
        for node in node_by_id.values():
            if node.get("kind") != "chunk":
                continue
            chunk_path = node.get("section_path") or []
            if chunk_path and section_title and chunk_path[-1] == section_title:
                matched.append(node["id"])

    deduped = []
    seen: set[str] = set()
    for chunk_id in matched:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(chunk_id)
        if len(deduped) == per_section:
            break
    return deduped


def concept_ids_for_chunks(
    chunk_ids: list[str],
    node_by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    max_per_chunk: int,
) -> tuple[dict[str, list[str]], list[str]]:
    chunk_to_concepts: dict[str, list[str]] = {}
    all_concepts: list[str] = []

    for chunk_id in chunk_ids:
        concept_ids: list[str] = []
        for edge in edges:
            if edge.get("relation") != "mentions":
                continue
            if edge.get("source") != chunk_id:
                continue
            target = edge.get("target")
            if target in node_by_id and node_by_id[target].get("kind") == "concept":
                concept_ids.append(target)

        if not concept_ids:
            # Fallback: sample from global concepts so the demo still renders.
            candidates = [node_id for node_id, node in node_by_id.items() if node.get("kind") == "concept"]
            random.shuffle(candidates)
            concept_ids = candidates[:max_per_chunk]

        unique_ids: list[str] = []
        seen: set[str] = set()
        for concept_id in concept_ids:
            if concept_id in seen:
                continue
            seen.add(concept_id)
            unique_ids.append(concept_id)
            if len(unique_ids) == max_per_chunk:
                break

        chunk_to_concepts[chunk_id] = unique_ids
        all_concepts.extend(unique_ids)

    deduped_all: list[str] = []
    seen_all: set[str] = set()
    for concept_id in all_concepts:
        if concept_id in seen_all:
            continue
        seen_all.add(concept_id)
        deduped_all.append(concept_id)
    return chunk_to_concepts, deduped_all


def pick_topic_records(topics_payload: dict[str, Any], selected_concept_ids: list[str], max_topics: int = 3) -> list[dict[str, Any]]:
    topics = topics_payload.get("topics")
    if not isinstance(topics, list):
        return []

    selected_set = set(selected_concept_ids)
    scored: list[tuple[int, dict[str, Any]]] = []
    for topic in topics:
        concept_ids = topic.get("concept_ids") or []
        overlap = len(selected_set.intersection(concept_ids))
        if overlap > 0:
            scored.append((overlap, topic))

    if not scored:
        return topics[:max_topics]

    scored.sort(key=lambda item: item[0], reverse=True)
    return [topic for _, topic in scored[:max_topics]]


def _draw_circle(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int, fill: str) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline="#ffffff", width=2)


def _draw_edges(
    draw: ImageDraw.ImageDraw,
    edges: list[tuple[str, str]],
    positions: dict[str, tuple[int, int]],
    color: str,
    width: int,
) -> None:
    for source, target in edges:
        source_pos = positions.get(source)
        target_pos = positions.get(target)
        if source_pos is None or target_pos is None:
            continue
        draw.line((source_pos[0], source_pos[1], target_pos[0], target_pos[1]), fill=color, width=width)


def _draw_legend(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> None:
    x0, y0 = 36, 28
    box_width = 260
    box_height = 34 + 30 * len(NODE_COLORS)
    draw.rounded_rectangle((x0, y0, x0 + box_width, y0 + box_height), radius=12, fill="#ffffff", outline="#d6dbe1", width=2)
    draw.text((x0 + 16, y0 + 10), "Node Types", fill="#1c2430", font=font)

    y = y0 + 42
    for kind, color in NODE_COLORS.items():
        _draw_circle(draw, (x0 + 22, y + 6), 7, color)
        draw.text((x0 + 40, y - 2), kind.capitalize(), fill="#1c2430", font=font)
        y += 30


def render_simplified_figure(
    output_path: Path,
    document_id: str,
    section_ids: list[str],
    leaf_section_ids: list[str],
    leaf_chunks: dict[str, list[str]],
    chunk_to_concepts: dict[str, list[str]],
    concept_links: list[tuple[str, str]],
    topic_records: list[dict[str, Any]],
) -> None:
    width, height = 1800, 1200
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    positions: dict[str, tuple[int, int]] = {}

    positions[document_id] = (900, 270)

    section_positions = [(560, 440), (1240, 440), (560, 620), (1240, 620)]
    for section_id, pos in zip(section_ids, section_positions):
        positions[section_id] = pos

    all_chunk_ids = [chunk_id for section_id in leaf_section_ids for chunk_id in leaf_chunks.get(section_id, [])]
    for index, chunk_id in enumerate(all_chunk_ids):
        x = 360 + index * 360
        y = 810
        positions[chunk_id] = (x, y)

    concept_ids: list[str] = []
    for chunk_id in all_chunk_ids:
        concept_ids.extend(chunk_to_concepts.get(chunk_id, []))
    concept_ids = list(dict.fromkeys(concept_ids))

    for index, concept_id in enumerate(concept_ids):
        x = 260 + index * 145
        y = 1030
        positions[concept_id] = (min(x, 1680), y)

    topic_ids: list[str] = []
    for index, topic in enumerate(topic_records):
        topic_id = topic.get("id", f"topic::{index}")
        topic_ids.append(topic_id)
        x = 700 + index * 180
        y = 100
        positions[topic_id] = (x, y)

    doc_edges = [(document_id, section_id) for section_id in section_ids]
    section_edges = [
        (section_ids[0], section_ids[1]),
        (section_ids[0], section_ids[2]),
        (section_ids[1], section_ids[3]),
    ]
    section_chunk_edges = [
        (section_id, chunk_id)
        for section_id in leaf_section_ids
        for chunk_id in leaf_chunks.get(section_id, [])
    ]
    chunk_concept_edges = [
        (chunk_id, concept_id)
        for chunk_id in all_chunk_ids
        for concept_id in chunk_to_concepts.get(chunk_id, [])
    ]
    topic_concept_edges: list[tuple[str, str]] = []
    concept_set = set(concept_ids)
    for idx, topic in enumerate(topic_records):
        topic_id = topic_ids[idx]
        for concept_id in topic.get("concept_ids", [])[:3]:
            if concept_id in concept_set:
                topic_concept_edges.append((topic_id, concept_id))

    _draw_edges(draw, doc_edges, positions, color="#7f8fa3", width=4)
    _draw_edges(draw, section_edges, positions, color="#93a4b7", width=3)
    _draw_edges(draw, section_chunk_edges, positions, color="#9ab2c6", width=3)
    _draw_edges(draw, chunk_concept_edges, positions, color="#9ab2c6", width=2)
    _draw_edges(draw, concept_links, positions, color="#d78a2c", width=2)
    _draw_edges(draw, topic_concept_edges, positions, color="#6baf5a", width=2)

    _draw_circle(draw, positions[document_id], radius=28, fill=NODE_COLORS["document"])
    for section_id in section_ids:
        _draw_circle(draw, positions[section_id], radius=22, fill=NODE_COLORS["section"])
    for chunk_id in all_chunk_ids:
        _draw_circle(draw, positions[chunk_id], radius=16, fill=NODE_COLORS["chunk"])
    for concept_id in concept_ids:
        _draw_circle(draw, positions[concept_id], radius=12, fill=NODE_COLORS["concept"])
    for topic_id in topic_ids:
        _draw_circle(draw, positions[topic_id], radius=16, fill=NODE_COLORS["topic"])

    _draw_legend(draw, font)
    image.save(output_path)


def build_demo_visualization(
    graph_payload: dict[str, Any],
    topics_payload: dict[str, Any],
    output_path: Path,
    chunks_per_leaf: int,
    concepts_per_chunk: int,
) -> None:
    random.seed(7)

    node_by_id, edges = build_indexes(graph_payload)
    document = pick_document(node_by_id)
    sections = pick_sections(document["id"], node_by_id, edges, max_sections=4)

    # Two sections are treated as leaves to make room for chunk and concept expansion.
    leaf_sections = sections[2:]
    leaf_chunks: dict[str, list[str]] = {
        leaf["id"]: chunk_ids_for_section(leaf, node_by_id, edges, chunks_per_leaf) for leaf in leaf_sections
    }
    selected_chunk_ids = [chunk_id for chunk_ids in leaf_chunks.values() for chunk_id in chunk_ids]

    chunk_to_concepts, selected_concept_ids = concept_ids_for_chunks(
        selected_chunk_ids,
        node_by_id,
        edges,
        concepts_per_chunk,
    )
    topic_records = pick_topic_records(topics_payload, selected_concept_ids, max_topics=2)

    selected_concept_set = set(selected_concept_ids)
    concept_related_edges = [
        edge
        for edge in edges
        if edge.get("relation") in {"related_to", "semantically_similar_to", "depends_on", "part_of"}
        and edge.get("source") in selected_concept_set
        and edge.get("target") in selected_concept_set
    ]
    concept_links = [(edge["source"], edge["target"]) for edge in concept_related_edges[:5]]

    if not concept_links:
        concept_list = list(selected_concept_set)
        for source_id, target_id in zip(concept_list[:-1], concept_list[1:]):
            concept_links.append((source_id, target_id))
            if len(concept_links) == 4:
                break

    render_simplified_figure(
        output_path=output_path,
        document_id=document["id"],
        section_ids=[section["id"] for section in sections],
        leaf_section_ids=[section["id"] for section in leaf_sections],
        leaf_chunks=leaf_chunks,
        chunk_to_concepts=chunk_to_concepts,
        concept_links=concept_links,
        topic_records=topic_records,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a simplified, presentation-friendly knowledge graph structure figure."
    )
    parser.add_argument(
        "--graph-json",
        type=Path,
        default=Path("outputs/20260425_224341_200401-marxist-Dien-Bien-Phu_a62b57/graph/graph_consolidated.json"),
        help="Path to graph_consolidated.json",
    )
    parser.add_argument(
        "--topics-json",
        type=Path,
        default=Path("outputs/20260425_224341_200401-marxist-Dien-Bien-Phu_a62b57/graph/topics.json"),
        help="Path to topics.json",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("outputs/kg_structure_demo.png"),
        help="Path to output figure (PNG)",
    )
    parser.add_argument(
        "--chunks-per-leaf",
        type=int,
        default=1,
        help="Number of chunks to attach to each leaf section",
    )
    parser.add_argument(
        "--concepts-per-chunk",
        type=int,
        default=2,
        help="Maximum number of concepts to attach per chunk",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph_json_path = args.graph_json
    topics_json_path = args.topics_json
    output_figure_path = args.output_figure

    if not graph_json_path.exists():
        raise FileNotFoundError(f"Graph JSON not found: {graph_json_path}")
    if not topics_json_path.exists():
        raise FileNotFoundError(f"Topics JSON not found: {topics_json_path}")

    graph_payload = load_json(graph_json_path)
    topics_payload = load_json(topics_json_path)

    output_figure_path.parent.mkdir(parents=True, exist_ok=True)
    build_demo_visualization(
        graph_payload=graph_payload,
        topics_payload=topics_payload,
        output_path=output_figure_path,
        chunks_per_leaf=max(1, int(args.chunks_per_leaf)),
        concepts_per_chunk=max(1, int(args.concepts_per_chunk)),
    )

    print(f"Saved simplified KG structure figure to: {output_figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
