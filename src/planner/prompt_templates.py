from __future__ import annotations

import json

from src.question_formats import QuestionFormatProfile, default_format_profile


def _legacy_format_rubric(profile: QuestionFormatProfile, num_questions: int) -> str:
    if profile.is_mcq_only():
        return (
            '- Every question_type must be "multiple_choice". '
            "The generator will produce four options; the plan only needs objectives and image hints."
        )
    counts = profile.expected_counts(num_questions)
    lines = [
        "Question format targets (approximate counts):",
    ]
    for fmt in sorted(profile.distribution.keys(), key=lambda x: (-profile.distribution[x], x)):
        lines.append(f"  - {fmt}: ~{counts[fmt]} of {num_questions}")
    lines.append(
        f"Use only these question_type values: {', '.join(sorted(profile.allowed_types))}."
    )
    return "\n".join(lines)


PLAN_TEMPLATE = """You are a quiz planning agent.

Goal:
- Build exactly {num_questions} quiz plan items from the supplied knowledge graph context.

Hard constraints:
- Use diverse target concepts whenever possible.
- Follow the target difficulty distribution: {difficulty_distribution}.
- reasoning_type must be one of: factoid, causal, multi-hop.
- Every planned question must include an associated image: provide `image_role` (illustrative, reasoning, distractor) and a concrete `image_description` suitable for generating an educational image.
- Question formats:
{format_rubric}

Output format:
- Return JSON only.
- Root object must contain key "questions".
- "questions" must be a list with exactly {num_questions} items.

- Each question object must include the following fields with exact types:
	- "target_concept": string
	- "question_type": string (e.g. "multiple_choice", "short_answer", "true_false")
	- "difficulty": string (one of "easy", "medium", "hard")
	- "reasoning_type": string (one of "factoid", "causal", "multi-hop")
	- "image_role": string ("illustrative" or "reasoning"). Illustrative means just for visualization. Reasoning means the question requires analyzing the image to answer.
	- "image_description": string (non-empty)
	- "learning_objective": string
	- "metadata": object

- Return only valid JSON. Do not include any explanatory text, markdown, or commentary.


Example single question item (must follow this exact schema):

{{
	"target_concept": "test_input::concept::42::neural_networks",
	"question_type": "multiple_choice",
	"difficulty": "easy",
	"reasoning_type": "factoid",
	"image_role": "illustrative",
	"image_description": "A diagram showing interconnected layers of nodes representing a neural network structure.",
	"learning_objective": "Identify components of a simple neural network.",
	"metadata": {{}}
}}

Full output example (root object with the questions list):
{{
	"questions": [
		/* include exactly {{num_questions}} items matching the single question schema above */
	]
}}

Context:
{graph_context}
"""


def render_planner_prompt(
	*,
	graph_context: dict[str, object],
	num_questions: int,
	difficulty_distribution: dict[str, float],
	format_profile: QuestionFormatProfile | None = None,
) -> str:
	fp = format_profile or default_format_profile()
	rubric = _legacy_format_rubric(fp, num_questions)
	indented = "\n".join(f"\t{r}" for r in rubric.split("\n"))
	return PLAN_TEMPLATE.format(
		num_questions=num_questions,
		difficulty_distribution=json.dumps(difficulty_distribution, sort_keys=True),
		format_rubric=indented,
		graph_context=json.dumps(graph_context, indent=2, sort_keys=True),
	)
