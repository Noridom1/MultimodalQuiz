"""Prompt templates for topic-driven agentic quiz generation."""

from __future__ import annotations

import json
from typing import Optional

from src.knowledge.schema import TopicContext


TOPIC_PLAN_TEMPLATE = """You are a topic-driven quiz planning agent.

Goal:
- Generate exactly {num_questions} quiz plan items grounded in a specific topic and its associated concepts.
- Maximize coverage across the {num_concepts} associated concepts to ensure breadth.

Hard constraints:
- Every planned question must cite a tested_fact_block_id: refer to the exact block ID from the provided chunks (see "concept_chunks" section below).
- reasoning_type must be one of: factoid, causal, multi-hop.
- Follow the target difficulty distribution: {difficulty_distribution}.
- Every planned question must include an associated image: provide `image_role` and a concrete `image_description`.

Output format:
- Return JSON only.
- Root object must contain key "questions".
- "questions" must be a list with exactly {num_questions} items.

Each question object must include:
	- "target_concept": string (one of the associated concepts; prefer diversity)
	- "question_type": string (e.g. "multiple_choice", "short_answer", "true_false")
	- "difficulty": string (one of "easy", "medium", "hard")
	- "reasoning_type": string (one of "factoid", "causal", "multi-hop")
	- "image_role": string ("illustrative" or "reasoning")
	- "image_description": string (non-empty)
	- "learning_objective": string
	- "tested_fact_block_id": string (MANDATORY: the block ID from the chunks below that grounds this question)
	- "metadata": object

Example question item:

{{
	"target_concept": "{example_concept}",
	"question_type": "multiple_choice",
	"difficulty": "easy",
	"reasoning_type": "factoid",
	"image_role": "illustrative",
	"image_description": "A diagram illustrating the concept.",
	"learning_objective": "Understand the fundamental aspect of the concept.",
	"tested_fact_block_id": "block_xyz",
	"metadata": {{}}
}}

Full output (root object):
{{
	"questions": [
		/* exactly {num_questions} items matching the schema above */
	]
}}

---
Topic and Context:
- Topic: {topic_label} (ID: {topic_id})
- Associated Concepts: {num_concepts} total

Concept Summaries:
{concept_list}

Chunks (grounding facts for tested_fact_block_id):
{chunks_list}

Images:
{images_list}

---
Instructions:
1. Generate exactly {num_questions} questions.
2. Spread questions across the {num_concepts} concepts (aim for {questions_per_concept} per concept).
3. Every question MUST reference a tested_fact_block_id from the chunks above (copy the exact block ID).
4. Prefer higher-confidence chunks (EXTRACTED > INFERRED).
5. Return ONLY valid JSON with no markdown or commentary.
"""


def render_topic_plan_prompt(
	topic_context: TopicContext,
	num_questions: int,
	difficulty_distribution: Optional[dict[str, float]] = None,
	only_concepts: list[str] | None = None,
) -> str:
	"""Render a prompt for topic-driven question planning.
	
	Args:
		topic_context: The TopicContext containing concepts, chunks, and images.
		num_questions: Number of questions to generate.
		difficulty_distribution: Optional dict mapping difficulty to proportion (e.g., {"easy": 0.5, "medium": 0.3, "hard": 0.2}).
		
	Returns:
		Formatted prompt string.
	"""
	if difficulty_distribution is None:
		difficulty_distribution = {"easy": 0.4, "medium": 0.4, "hard": 0.2}

	# Build concept list (optionally filtered to only_concepts)
	concept_list = ""
	assoc = topic_context.associated_concepts
	if only_concepts is not None:
		assoc = [c for c in assoc if c.id in only_concepts or c.label in only_concepts]

	for i, concept in enumerate(assoc, 1):
		concept_text = f"- [{i}] {concept.label} (ID: {concept.id})"
		if concept.text:
			# Include first 100 chars of concept text if available
			truncated = concept.text[:100] + ("..." if len(concept.text) > 100 else "")
			concept_text += f"\n  Definition/Description: {truncated}"
		concept_list += concept_text + "\n"

	# Build chunks list with block IDs for citation
	chunks_list = ""
	# Filter chunks to only include those for concepts we're showing
	concept_chunks_items = topic_context.concept_chunks.items()
	if only_concepts is not None:
		concept_chunks_items = ((cid, chs) for cid, chs in topic_context.concept_chunks.items() if cid in only_concepts or cid in [c for c in only_concepts])

	for concept_id, chunks in concept_chunks_items:
		concept_label = next((c.label for c in topic_context.associated_concepts if c.id == concept_id), concept_id)
		chunks_list += f"\n[Concept: {concept_label}]\n"
		for chunk in chunks[:5]:  # Limit to top 5 chunks per concept
			# Truncate long text to 200 chars
			truncated_text = chunk.text[:200] + ("..." if len(chunk.text) > 200 else "")
			chunks_list += f"  - Block ID: {chunk.source_block_id}\n    Text: {truncated_text}\n"

	# Build images list
	images_list = ""
	image_items = topic_context.concept_images.items()
	if only_concepts is not None:
		image_items = ((cid, imgs) for cid, imgs in topic_context.concept_images.items() if cid in only_concepts or cid in [c for c in only_concepts])

	for concept_id, image_ids in image_items:
		if image_ids:
			concept_label = next((c.label for c in topic_context.associated_concepts if c.id == concept_id), concept_id)
			images_list += f"\n[Concept: {concept_label}]\n"
			for image_id in image_ids[:3]:  # Limit to top 3 images per concept
				images_list += f"  - {image_id}\n"

	if not images_list:
		images_list = "(No images available for this topic)"

	# Calculate questions per concept
	num_concepts = len(topic_context.associated_concepts)
	questions_per_concept = max(1, num_questions // num_concepts) if num_concepts > 0 else num_questions

	# Pick an example concept
	example_concept = (
		topic_context.associated_concepts[0].label
		if topic_context.associated_concepts
		else "concept"
	)

	return TOPIC_PLAN_TEMPLATE.format(
		num_questions=num_questions,
		num_concepts=num_concepts,
		topic_id=topic_context.topic_id,
		topic_label=topic_context.topic_label,
		difficulty_distribution=json.dumps(difficulty_distribution, sort_keys=True),
		concept_list=concept_list,
		chunks_list=chunks_list,
		images_list=images_list,
		questions_per_concept=questions_per_concept,
		example_concept=example_concept,
	)
