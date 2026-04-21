from __future__ import annotations

import json


PLAN_TEMPLATE = """You are a quiz planning agent.

Goal:
- Build exactly {num_questions} quiz plan items from the supplied knowledge graph context.

Hard constraints:
- Use diverse target concepts whenever possible.
- Follow the target difficulty distribution: {difficulty_distribution}.
- reasoning_type must be one of: factoid, causal, multi-hop.
- image_role must be one of: illustrative, reasoning, distractor when requires_image is true.
- If requires_image is false, image_role must be "none" and image_description must be null.
- If requires_image is true, image_description must be a short concrete visual description.

Output format:
- Return JSON only.
- Root object must contain key "questions".
- "questions" must be a list with exactly {num_questions} items.

- Each question object must include the following fields with exact types:
	- "target_concept": string
	- "question_type": string (e.g. "multiple_choice", "short_answer", "true_false")
	- "difficulty": string (one of "easy", "medium", "hard")
	- "reasoning_type": string (one of "factoid", "causal", "multi-hop")
	- "requires_image": boolean
	- "image_role": string or "none" (if requires_image true: "illustrative", "reasoning", "distractor")
	- "image_description": string or null (must be non-empty when requires_image is true)
	- "learning_objective": string
	- "metadata": object

- Return only valid JSON. Do not include any explanatory text, markdown, or commentary.

Example single question item (must follow this exact schema):

{{
	"target_concept": "test_input::concept::42::neural_networks",
	"question_type": "multiple_choice",
	"difficulty": "easy",
	"reasoning_type": "factoid",
	"requires_image": true,
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
) -> str:
	return PLAN_TEMPLATE.format(
		num_questions=num_questions,
		difficulty_distribution=json.dumps(difficulty_distribution, sort_keys=True),
		graph_context=json.dumps(graph_context, indent=2, sort_keys=True),
	)
