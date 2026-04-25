"""Topic-driven context retriever for graph traversal and budget allocation."""

from __future__ import annotations

from typing import Optional

from src.knowledge.schema import (
    ConceptNode,
    EdgeRelation,
    MultimodalDocumentGraph,
    NodeKind,
    TextChunk,
    TopicContext,
)


class TopicContextRetriever:
    """Retrieves rich context for a topic node from the MultimodalDocumentGraph.
    
    Traverses from a topic node to its associated concepts, text chunks, and images,
    applying strict confidence filtering to ensure only high-quality extracted artifacts
    are included.
    """

    def __init__(self, graph: MultimodalDocumentGraph):
        """Initialize retriever with a MultimodalDocumentGraph.
        
        Args:
            graph: The MultimodalDocumentGraph to retrieve context from.
        """
        self.graph = graph
        # Build lookup dicts for fast access
        self._nodes_by_id = {node.id: node for node in graph.nodes}
        self._edges_by_source = {}
        self._edges_by_target = {}
        
        for edge in graph.edges:
            if edge.source not in self._edges_by_source:
                self._edges_by_source[edge.source] = []
            self._edges_by_source[edge.source].append(edge)
            
            if edge.target not in self._edges_by_target:
                self._edges_by_target[edge.target] = []
            self._edges_by_target[edge.target].append(edge)

    def retrieve_context(self, topic_id: str) -> TopicContext:
        """Retrieve rich context for a topic node.
        
        Args:
            topic_id: The ID of the topic node.
            
        Returns:
            TopicContext containing associated concepts, chunks, and images.
            
        Raises:
            ValueError: If topic_id is not found or is not a topic node.
        """
        topic_node = self._nodes_by_id.get(topic_id)
        if not topic_node:
            raise ValueError(f"Topic node {topic_id} not found in graph")
        if topic_node.kind != NodeKind.concept:  # Topics are typically concept nodes
            raise ValueError(f"Node {topic_id} is not a concept (kind={topic_node.kind})")

        # Initialize context
        context = TopicContext(
            topic_id=topic_id,
            topic_label=topic_node.label,
        )

        # Find associated concepts via 'groups' or 'mentions' edges
        associated_concepts = self._find_associated_concepts(topic_id)
        context.associated_concepts = associated_concepts

        # For each concept, find chunks and images
        concept_chunks = {}
        concept_images = {}
        
        for concept in associated_concepts:
            chunks = self._find_chunks_for_concept(concept.id)
            images = self._find_images_for_concept(concept.id)
            
            concept_chunks[concept.id] = chunks
            concept_images[concept.id] = images

        context.concept_chunks = concept_chunks
        context.concept_images = concept_images

        # Calculate totals
        context.total_chunk_count = sum(len(chunks) for chunks in concept_chunks.values())
        context.total_image_count = sum(len(images) for images in concept_images.values())
        
        # Estimate questions based on context density
        # Heuristic: 1 question per 2 chunks, or 1 per unique concept, whichever is higher
        context.estimated_questions = max(
            (context.total_chunk_count // 2) + 1,
            len(associated_concepts)
        )

        return context

    def _find_associated_concepts(self, topic_id: str) -> list[ConceptNode]:
        """Find all concept nodes associated with a topic via 'groups' or 'mentions' edges.
        
        Args:
            topic_id: The topic node ID.
            
        Returns:
            List of ConceptNode objects.
        """
        concepts = []
        visited = set()

        # Use groups and mentions edges from the topic
        edge_types = {EdgeRelation.mentions, EdgeRelation.defines, EdgeRelation.explains}
        
        # BFS to find connected concepts
        queue = [topic_id]
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            # Find outgoing edges
            outgoing_edges = self._edges_by_source.get(current_id, [])
            for edge in outgoing_edges:
                if edge.confidence != "EXTRACTED":
                    # Skip non-extracted edges for strict filtering
                    continue
                
                target_node = self._nodes_by_id.get(edge.target)
                if not target_node:
                    continue

                # If target is a concept, add it
                if target_node.kind == NodeKind.concept:
                    if target_node.id not in visited:
                        concepts.append(ConceptNode(
                            id=target_node.id,
                            label=target_node.label,
                            text=target_node.text,
                            metadata=target_node.metadata,
                        ))
                        queue.append(target_node.id)

        return concepts

    def _find_chunks_for_concept(self, concept_id: str) -> list[TextChunk]:
        """Find all text chunks associated with a concept.
        
        Traverses via 'mentions' and 'references' edges to find block/section nodes,
        then extracts text as chunks.
        
        Args:
            concept_id: The concept node ID.
            
        Returns:
            List of TextChunk objects.
        """
        chunks = []
        visited_blocks = set()

        # Find blocks/sections connected to the concept
        edge_types = {EdgeRelation.mentions, EdgeRelation.references, EdgeRelation.supports}
        
        outgoing = self._edges_by_source.get(concept_id, [])
        for edge in outgoing:
            if edge.confidence != "EXTRACTED":
                continue
            if edge.relation not in edge_types:
                continue

            block_node = self._nodes_by_id.get(edge.target)
            if not block_node:
                continue
            if block_node.kind not in {NodeKind.block, NodeKind.section}:
                continue
            if block_node.id in visited_blocks:
                continue

            visited_blocks.add(block_node.id)
            
            # Create a text chunk from this block
            if block_node.text:
                chunk = TextChunk(
                    id=f"{concept_id}_chunk_{len(chunks)}",
                    text=block_node.text,
                    source_block_id=block_node.id,
                    section_path=block_node.section_path,
                    confidence="EXTRACTED",
                    metadata={
                        "source_label": block_node.label,
                        "source_file": block_node.source_file,
                    }
                )
                chunks.append(chunk)

        return chunks

    def _find_images_for_concept(self, concept_id: str) -> list[str]:
        """Find all image IDs associated with a concept.
        
        Traverses via 'illustrates' and related edges to find image nodes.
        
        Args:
            concept_id: The concept node ID.
            
        Returns:
            List of image node IDs.
        """
        image_ids = []
        visited = set()

        # Find blocks connected to the concept
        outgoing = self._edges_by_source.get(concept_id, [])
        block_ids = []
        
        for edge in outgoing:
            if edge.confidence != "EXTRACTED":
                continue
            if edge.relation not in {EdgeRelation.mentions, EdgeRelation.references}:
                continue

            block_node = self._nodes_by_id.get(edge.target)
            if block_node and block_node.kind in {NodeKind.block, NodeKind.section}:
                block_ids.append(block_node.id)

        # For each block, find connected images
        for block_id in block_ids:
            block_edges = self._edges_by_source.get(block_id, [])
            for edge in block_edges:
                if edge.confidence != "EXTRACTED":
                    continue
                if edge.relation != EdgeRelation.illustrates:
                    continue

                image_node = self._nodes_by_id.get(edge.target)
                if image_node and image_node.kind == NodeKind.image:
                    if image_node.id not in visited:
                        visited.add(image_node.id)
                        image_ids.append(image_node.id)

        return image_ids


def calculate_topic_budget(
    topic_context: TopicContext,
    remaining_budget: int,
    total_topic_count: int,
    total_resources: int,
    max_per_topic: int = 10,
) -> int:
    """Calculate proportional question budget allocation for a topic.
    
    Uses a proportional allocation based on the topic's content density
    (concepts + chunks) relative to total resources across all topics.
    
    Args:
        topic_context: The TopicContext for the topic.
        remaining_budget: Remaining question count to allocate.
        total_topic_count: Total number of topics in the graph.
        total_resources: Total content resources (chunks + concepts) across all topics.
        max_per_topic: Maximum questions to allocate to a single topic.
        
    Returns:
        Number of questions (0 to max_per_topic) allocated to this topic.
    """
    if remaining_budget <= 0 or total_resources <= 0:
        return 0

    # Calculate topic's resource proportion
    topic_resources = topic_context.total_chunk_count + len(topic_context.associated_concepts)
    
    if topic_resources == 0:
        # Empty topic gets 0 questions
        return 0

    # Proportional share of remaining budget
    proportional_share = topic_resources / total_resources
    allocated = int(remaining_budget * proportional_share)

    # Cap by max_per_topic and ensure at least 1 if we have content
    allocated = max(1, min(allocated, remaining_budget, max_per_topic))

    return allocated
