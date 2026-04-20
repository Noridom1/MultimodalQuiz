from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KnowledgeNode:
    id: str
    label: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class KnowledgeEdge:
    source: str
    target: str
    relation: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    nodes: list[KnowledgeNode] = field(default_factory=list)
    edges: list[KnowledgeEdge] = field(default_factory=list)
