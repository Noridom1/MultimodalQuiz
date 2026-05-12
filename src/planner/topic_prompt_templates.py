"""Prompt templates for topic-driven agentic quiz generation."""

from __future__ import annotations

import json
from typing import Optional

from src.knowledge.schema import TopicContext
from src.question_formats import QuestionFormatProfile


def _distribution_rubric(profile: QuestionFormatProfile, num_questions: int) -> str:
    lines = [
        "Question format targets (approximate counts for this batch):",
    ]
    counts = profile.expected_counts(num_questions)
    for fmt, weight in sorted(profile.distribution.items(), key=lambda x: (-x[1], x[0])):
        n = counts.get(fmt, 0)
        lines.append(f"  - {fmt}: ~{n} of {num_questions} questions (weight {weight:.2f})")
    lines.append(
        f"Use only these question_type values: {', '.join(sorted(profile.allowed_types))}."
    )
    return "\n".join(lines)


def _per_type_schema_bullets(profile: QuestionFormatProfile, num_options: int) -> str:
    common = """\t- "target_concept": string (one of the associated concepts; prefer diversity)
\t- "question_type": string (one of the allowed values below)
\t- "difficulty": string (one of "easy", "medium", "hard")
\t- "reasoning_type": string (one of "factoid", "causal", "multi-hop")
\t- "image_role": string ("illustrative" or "reasoning")
\t- "image_description": string (non-empty)
\t- "learning_objective": string
\t- "tested_fact_block_id": string (MANDATORY: block ID from the chunks below)
\t- "metadata": object"""

    extra: list[str] = []
    if "multiple_choice" in profile.allowed_types:
        extra.append(
            f'\nIf question_type is "multiple_choice", also include:\n'
            f'\t- "options": list of strings (exactly {num_options} choices)\n'
            f'\t- "correct_option_index": integer (0-based index into `options`)'
        )
    if "true_false" in profile.allowed_types:
        extra.append(
            '\nIf question_type is "true_false", optionally include '
            '`"options": ["True", "False"]` and `"correct_option_index"` (0=True, 1=False); '
            "these may be omitted because the generator will finalize wording."
        )
    if "fill_in_blank" in profile.allowed_types:
        extra.append(
            '\nIf question_type is "fill_in_blank", no extra keys are required at plan time; '
            "the generator will produce the blanked stem and answer."
        )
    if "matching" in profile.allowed_types:
        extra.append(
            '\nIf question_type is "matching", put a draft pairing list in metadata only, e.g. '
            '`"metadata": {"matching_pairs": [{"left": "...", "right": "..."}, ...]}` '
            "with at least 3 pairs grounded in chunks (generator will polish)."
        )
    return common + "".join(extra)


def _format_constraints_block(
    profile: QuestionFormatProfile,
    num_questions: int,
    num_options: int,
    difficulty_distribution: dict[str, float],
    required_types: list[str] | None = None,
) -> str:
    dd = json.dumps(difficulty_distribution, sort_keys=True)
    lines = [
        "- Every planned question must cite a tested_fact_block_id: refer to the exact block ID from the provided chunks (see \"concept_chunks\" section below).",
        "- reasoning_type must be one of: factoid, causal, multi-hop.",
        f"- Follow the target difficulty distribution: {dd}.",
        "- Every planned question must include an associated image: provide `image_role` and a concrete `image_description`.",
    ]
    if profile.is_mcq_only():
        lines.append(f"- Every planned question MUST be multiple-choice with exactly {num_options} options.")
    else:
        for rubric_line in _distribution_rubric(profile, num_questions).split("\n"):
            lines.append(f"- {rubric_line}")
    if required_types:
        lines.append(f"- For this batch, use this exact question_type sequence by index: {json.dumps(required_types)}.")
    return "\n".join(lines)


def _tail_instructions(profile: QuestionFormatProfile, num_questions: int, num_concepts: int) -> str:
    lines = [
        f"1. Generate exactly {num_questions} questions.",
        f"2. Spread questions across the {num_concepts} concepts (aim for {{questions_per_concept}} per concept).",
        "3. Every question MUST reference a tested_fact_block_id from the chunks above (copy the exact block ID).",
        "4. Prefer higher-confidence chunks (EXTRACTED > INFERRED).",
        "5. Return ONLY valid JSON with no markdown or commentary.",
    ]
    n = 6
    if profile.is_mcq_only():
        lines.append(
            "6. For each multiple-choice question, ensure exactly one option is correct and set "
            "`correct_option_index` accordingly."
        )
        lines.append("7. Distractor quality rules (important):")
        lines.append(
            '\t- Definition: "distractors" are the non-correct options (index != `correct_option_index`).'
        )
        lines.append(
            "\t- Distractors must be plausible; avoid trivially wrong or off-domain options."
        )
        lines.append(
            "\t- Prefer misconceptions or close alternatives from chunks; no duplicate option strings."
        )
        lines.append(
            "\t- For numeric items, use distractors close in magnitude when appropriate."
        )
        lines.append(
            "8. When possible, trace distractors to chunks; options must not contradict grounding."
        )
        lines.append("9. Return only the JSON object described above.")
        return "\n".join(lines)

    lines.append(
        f"{n}. Match the format targets above as closely as possible (approximate counts per question_type)."
    )
    n += 1
    lines.append(
        f"{n}. For multiple_choice rows, follow the same distractor quality rules as in the MCQ-only pipeline."
    )
    n += 1
    lines.append(f"{n}. Return only the JSON object described above.")
    return "\n".join(lines)


TOPIC_PLAN_TEMPLATE = """You are a topic-driven quiz planning agent.

Goal:
- Generate exactly {num_questions} quiz plan items grounded in a specific topic and its associated concepts.
- Maximize coverage across the {num_concepts} associated concepts to ensure breadth.

Hard constraints:
{format_constraints}

Output format:
- Return JSON only.
- Root object must contain key "questions".
- "questions" must be a list with exactly {num_questions} items.

Each question object must include:
{per_type_schema}

Example question item:
{example_question}

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
{tail_instructions}
"""


def _build_example_question(profile: QuestionFormatProfile, example_concept: str, num_options: int) -> str:
    if profile.is_mcq_only():
        return f"""{{
	"target_concept": "{example_concept}",
	"question_type": "multiple_choice",
	"options": ["Option A", "Option B", "Option C", "Option D"],
	"correct_option_index": 0,
	"difficulty": "easy",
	"reasoning_type": "factoid",
	"image_role": "illustrative",
	"image_description": "A diagram illustrating the concept.",
	"learning_objective": "Understand the fundamental aspect of the concept.",
	"tested_fact_block_id": "block_xyz",
	"metadata": {{}}
}}"""
    # Mixed: show MCQ if allowed, else first allowed type
    if "multiple_choice" in profile.allowed_types:
        return f"""{{
	"target_concept": "{example_concept}",
	"question_type": "multiple_choice",
	"options": ["Option A", "Option B", "Option C", "Option D"],
	"correct_option_index": 0,
	"difficulty": "easy",
	"reasoning_type": "factoid",
	"image_role": "illustrative",
	"image_description": "A diagram illustrating the concept.",
	"learning_objective": "Understand the fundamental aspect of the concept.",
	"tested_fact_block_id": "block_xyz",
	"metadata": {{}}
}}"""
    if "true_false" in profile.allowed_types:
        return f"""{{
	"target_concept": "{example_concept}",
	"question_type": "true_false",
	"difficulty": "easy",
	"reasoning_type": "factoid",
	"image_role": "illustrative",
	"image_description": "A simple illustration related to the fact.",
	"learning_objective": "Verify understanding of a factual claim.",
	"tested_fact_block_id": "block_xyz",
	"metadata": {{}}
}}"""
    if "fill_in_blank" in profile.allowed_types:
        return f"""{{
	"target_concept": "{example_concept}",
	"question_type": "fill_in_blank",
	"difficulty": "medium",
	"reasoning_type": "factoid",
	"image_role": "illustrative",
	"image_description": "Context illustration for the term to recall.",
	"learning_objective": "Recall a key term from the material.",
	"tested_fact_block_id": "block_xyz",
	"metadata": {{}}
}}"""
    return f"""{{
	"target_concept": "{example_concept}",
	"question_type": "matching",
	"difficulty": "medium",
	"reasoning_type": "factoid",
	"image_role": "illustrative",
	"image_description": "Labeled items that correspond to definitions.",
	"learning_objective": "Associate terms with their meanings.",
	"tested_fact_block_id": "block_xyz",
	"metadata": {{
		"matching_pairs": [
			{{"left": "Term A", "right": "Definition A"}},
			{{"left": "Term B", "right": "Definition B"}},
			{{"left": "Term C", "right": "Definition C"}}
		]
	}}
}}"""


def render_topic_plan_prompt(
    topic_context: TopicContext,
    num_questions: int,
    difficulty_distribution: Optional[dict[str, float]] = None,
    only_concepts: list[str] | None = None,
    num_options: int = 4,
    format_profile: QuestionFormatProfile | None = None,
    required_types: list[str] | None = None,
) -> str:
    """Render a prompt for topic-driven question planning."""
    if difficulty_distribution is None:
        difficulty_distribution = {"easy": 0.4, "medium": 0.4, "hard": 0.2}

    if format_profile is None:
        format_profile = QuestionFormatProfile(distribution={"multiple_choice": 1.0})

    concept_list = ""
    assoc = topic_context.associated_concepts
    if only_concepts is not None:
        assoc = [c for c in assoc if c.id in only_concepts or c.label in only_concepts]

    for i, concept in enumerate(assoc, 1):
        concept_text = f"- [{i}] {concept.label} (ID: {concept.id})"
        if concept.text:
            truncated = concept.text[:100] + ("..." if len(concept.text) > 100 else "")
            concept_text += f"\n  Definition/Description: {truncated}"
        concept_list += concept_text + "\n"

    chunks_list = ""
    concept_chunks_items = topic_context.concept_chunks.items()
    if only_concepts is not None:
        concept_chunks_items = (
            (cid, chs)
            for cid, chs in topic_context.concept_chunks.items()
            if cid in only_concepts or cid in [c for c in only_concepts]
        )

    for concept_id, chunks in concept_chunks_items:
        concept_label = next(
            (c.label for c in topic_context.associated_concepts if c.id == concept_id),
            concept_id,
        )
        chunks_list += f"\n[Concept: {concept_label}]\n"
        for chunk in chunks[:5]:
            truncated_text = chunk.text[:200] + ("..." if len(chunk.text) > 200 else "")
            chunks_list += f"  - Block ID: {chunk.source_block_id}\n    Text: {truncated_text}\n"

    images_list = ""
    image_items = topic_context.concept_images.items()
    if only_concepts is not None:
        image_items = (
            (cid, imgs)
            for cid, imgs in topic_context.concept_images.items()
            if cid in only_concepts or cid in [c for c in only_concepts]
        )

    for concept_id, image_ids in image_items:
        if image_ids:
            concept_label = next(
                (c.label for c in topic_context.associated_concepts if c.id == concept_id),
                concept_id,
            )
            images_list += f"\n[Concept: {concept_label}]\n"
            for image_id in image_ids[:3]:
                images_list += f"  - {image_id}\n"

    if not images_list:
        images_list = "(No images available for this topic)"

    num_concepts = len(topic_context.associated_concepts)
    questions_per_concept = max(1, num_questions // num_concepts) if num_concepts > 0 else num_questions

    example_concept = (
        topic_context.associated_concepts[0].label
        if topic_context.associated_concepts
        else "concept"
    )

    fc_block = _format_constraints_block(
        format_profile, num_questions, num_options, difficulty_distribution, required_types
    )
    per_type = _per_type_schema_bullets(format_profile, num_options)
    example_q = _build_example_question(format_profile, example_concept, num_options)
    tail = _tail_instructions(format_profile, num_questions, num_concepts).format(
        questions_per_concept=questions_per_concept,
    )

    return TOPIC_PLAN_TEMPLATE.format(
        num_questions=num_questions,
        num_concepts=num_concepts,
        topic_id=topic_context.topic_id,
        topic_label=topic_context.topic_label,
        concept_list=concept_list,
        chunks_list=chunks_list,
        images_list=images_list,
        format_constraints=fc_block,
        per_type_schema=per_type,
        example_question=example_q,
        tail_instructions=tail,
    )
