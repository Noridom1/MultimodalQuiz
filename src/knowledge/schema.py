from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

import networkx as nx
from pydantic import BaseModel, Field


class NodeKind(str, Enum):
    block = "block"
    concept = "concept"
    image = "image"
    section = "section"


class EdgeRelation(str, Enum):
    contains = "contains"
    follows = "follows"
    mentions = "mentions"
    references = "references"
    explains = "explains"
    illustrates = "illustrates"
    supports = "supports"
    defines = "defines"
    related_to = "related_to"
    semantically_similar_to = "semantically_similar_to"


class GraphNode(BaseModel):
    id: str
    label: str
    kind: NodeKind
    source_file: Optional[str] = None
    page_idx: Optional[int] = None
    section_path: list[str] = Field(default_factory=list)
    text: Optional[str] = None
    image_path: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: EdgeRelation
    confidence: Literal["EXTRACTED", "INFERRED", "AMBIGUOUS"] = "EXTRACTED"
    confidence_score: float = 1.0
    weight: float = 1.0
    source_file: Optional[str] = None
    source_location: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MultimodalDocumentGraph(BaseModel):
    document_id: str
    source_file: Optional[str] = None
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_networkx(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        for node in self.nodes:
            graph.add_node(node.id, **node.model_dump())

        for edge in self.edges:
            graph.add_edge(
                edge.source,
                edge.target,
                key=f"{edge.relation.value}:{edge.source}->{edge.target}",
                **edge.model_dump(),
            )

        graph.graph.update(self.metadata)
        graph.graph["document_id"] = self.document_id
        graph.graph["source_file"] = self.source_file
        return graph

    def summary(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_file": self.source_file,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_kinds": _count_enum_values(node.kind for node in self.nodes),
            "edge_relations": _count_enum_values(edge.relation for edge in self.edges),
        }


def _count_enum_values(values: list[Enum] | tuple[Enum, ...] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value.value if isinstance(value, Enum) else str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def make_document_id(source_file: str | Path) -> str:
    return Path(source_file).stem


class Question(BaseModel):
    id: str
    question_text: str
    options: list[str] = Field(default_factory=list)
    correct_answer: str
    explanation: str
    target_concept: str
    difficulty: str
    question_type: str
    associated_image: Optional[str] = None
    image_grounded: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def validate(self) -> None:
        """Basic validation for question payloads used by generators.

        Raises ValueError on invalid content.
        """
        if not self.question_text or not self.question_text.strip():
            raise ValueError("question_text must be non-empty")

        if self.question_type == "multiple-choice" or self.question_type == "multiple_choice":
            if not isinstance(self.options, list) or len(self.options) != 4:
                raise ValueError("multiple-choice questions must include 4 options")
            opts = [str(o).strip() for o in self.options]
            if not all(opts):
                raise ValueError("options must be non-empty strings")
            ca = str(self.correct_answer).strip()
            # allow correct_answer as option text or label A/B/C/D
            if ca not in opts and ca.upper() not in {"A", "B", "C", "D"}:
                raise ValueError("correct_answer must match one of the options or A/B/C/D")

        if not self.difficulty or self.difficulty.lower() not in {"easy", "medium", "hard"}:
            raise ValueError("difficulty must be one of: easy, medium, hard")

        if self.associated_image is None or not str(self.associated_image).strip():
            raise ValueError("associated_image must be a non-empty URL/path for multimodal questions")


class TextChunk(BaseModel):
    """Represents a text chunk extracted from a block or section node."""
    id: str
    text: str
    source_block_id: str
    section_path: list[str] = Field(default_factory=list)
    confidence: Literal["EXTRACTED", "INFERRED", "AMBIGUOUS"] = "EXTRACTED"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConceptNode(BaseModel):
    """Represents a concept node from the graph with metadata."""
    id: str
    label: str
    text: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TopicContext(BaseModel):
    """Rich context for a topic node, including associated concepts and artifacts."""
    topic_id: str
    topic_label: str
    associated_concepts: list[ConceptNode] = Field(default_factory=list)
    concept_chunks: dict[str, list[TextChunk]] = Field(default_factory=dict)  # concept_id -> chunks
    concept_images: dict[str, list[str]] = Field(default_factory=dict)  # concept_id -> image ids
    total_chunk_count: int = 0
    total_image_count: int = 0
    estimated_questions: int = 0  # Computed from context density
