from __future__ import annotations

import json

from src.planner.planner import QuestionPlan

def build_question_prompt(question_plan: QuestionPlan) -> str:
    """Build the prompt used to generate a question from a plan."""
    context = question_plan.metadata.get("knowledge_context", "")
    if not isinstance(context, str):
        context = ""

    schema = {
        "question_text": "string",
        "options": ["string"],
        "correct_answer": "string",
        "explanation": "string",
    }

    image_rule = (
        "The question must require visual evidence from the associated image. "
        "Reference visible attributes such as shape, labels, orientation, relative position, color, "
        "or annotations in the explanation."
        if question_plan.requires_image
        else "The question should be answerable from text context without image dependency."
    )

    constraints = [
        "Return ONLY valid JSON with no markdown fences.",
        "Do not include any keys beyond the required schema.",
        "If question_type is multiple-choice, include 4 options.",
        "Ensure correct_answer matches either the exact option text or option label (A/B/C/D).",
        "Keep explanation concise and instructional (1-3 sentences).",
        image_rule,
    ]

    prompt_lines = [
        "You are generating one high-quality quiz question.",
        f"Target concept: {question_plan.target_concept}",
        f"Question type: {question_plan.question_type}",
        f"Difficulty: {question_plan.difficulty}",
        f"Reasoning type: {question_plan.reasoning_type}",
    ]

    if question_plan.learning_objective:
        prompt_lines.append(f"Learning objective: {question_plan.learning_objective}")
    if question_plan.image_description:
        prompt_lines.append(f"Image context: {question_plan.image_description}")
    if context.strip():
        prompt_lines.append(f"Knowledge context: {context.strip()}")

    prompt_lines.extend(
        [
            "Required output JSON schema:",
            json.dumps(schema, ensure_ascii=True),
            "Constraints:",
            *[f"- {item}" for item in constraints],
        ]
    )

    return "\n".join(prompt_lines)


def build_image_prompt(question_plan: QuestionPlan) -> str:
    """Build the prompt used to generate an image for a question."""
    role = question_plan.image_role or "illustration"
    description = question_plan.image_description or question_plan.target_concept
    objective = question_plan.learning_objective or "support quiz question answering"

    return (
        f"Create a clear educational {role} about '{question_plan.target_concept}'. "
        f"It should depict: {description}. "
        f"The image must help learners {objective}. "
        "Use clean composition, legible labels, and avoid decorative clutter."
    )
    