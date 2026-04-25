from __future__ import annotations

import json
from typing import Callable, Optional, Dict, Any

from src.planner.planner import QuestionPlan

class PromptBuilder:
    """Build image and question prompts from a question plan."""

    @staticmethod
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

        image_role = (question_plan.image_role or "illustrative").strip().lower()
        if image_role == "reasoning":
            image_rule = (
                "The question must require visual evidence from the associated image. "
                "Reference visible attributes such as shape, labels, orientation, relative position, color, "
                "or annotations in the explanation."
            )
        else:
            image_rule = (
                "The associated image is illustrative support only. "
                "The question should remain answerable without inspecting the image."
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
            f"Image role: {image_role}",
        ]

        if question_plan.learning_objective:
            prompt_lines.append(f"Learning objective: {question_plan.learning_objective}")
        if question_plan.image_description:
            prompt_lines.append(f"Image context: {question_plan.image_description}")
        if question_plan.tested_fact_block_id:
            prompt_lines.append(f"Tested fact block ID: {question_plan.tested_fact_block_id}")
            prompt_lines.append("Cite this fact source in your explanation.")
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

    @staticmethod
    def build_image_prompt(question_plan: QuestionPlan) -> str:
        """Build the prompt used to generate an image for a question."""
        # Backwards-compatible wrapper: produce a concise instruction string
        role = question_plan.image_role or "illustration"
        description = question_plan.image_description or question_plan.target_concept
        objective = question_plan.learning_objective or "support quiz question answering"

        return (
            f"Create a clear educational {role} about '{question_plan.target_concept}'. "
            f"It should depict: {description}. "
            f"The image must help learners {objective}. "
            "Use clean composition, legible labels, and avoid decorative clutter."
        )

    @staticmethod
    def build_image_prompt_via_llm(
        question_plan: QuestionPlan,
        llm: Optional[Callable[[str], str]] = None,
    ) -> str:
        """Construct a detailed, structured prompt for an LLM to produce image-generation instructions.

        If `llm` is provided it will be called with the composed prompt and its output returned.
        Otherwise the composed LLM prompt string is returned for an external caller to send to an LLM.
        The prompt asks the LLM to return a concise JSON object describing the image and artist
        guidance including `description`, `composition`, `labels`, `color_palette`, `camera`,
        `annotations`, `alt_text`, and `notes_for_illustrator`.
        """
        role = (question_plan.image_role or "illustration").strip()
        description = question_plan.image_description or question_plan.target_concept
        objective = question_plan.learning_objective or "support quiz question answering"

        schema: Dict[str, Any] = {
            "description": "string (short, 1-2 sentences)",
            "composition": "string (what is shown, focal point, perspective)",
            "labels": ["string"],
            "color_palette": "string (2-4 main colors / tones)",
            "camera": "string (perspective/zoom, e.g. top-down, close-up)",
            "annotations": ["string"],
            "alt_text": "string (one-line accessibility description)",
            "notes_for_illustrator": "string (any extra constraints)",
        }

        constraints = [
            "Return ONLY valid JSON that matches the required schema.",
            "Keep `description` short (1-2 sentences) and `notes_for_illustrator` actionable.",
            "Avoid copyrighted characters, real brand logos, or trademarked designs.",
            "Prefer legible fonts, clear labels, and high contrast for educational clarity.",
            "Include an `alt_text` field suitable for screen readers (1 sentence).",
        ]

        prompt_lines = [
            "SYSTEM: You are an expert image-generation prompt engineer. Produce a compact JSON\ndescription that an illustrator or image model can follow to produce a clear educational image.",
            "USER:",
            f"Target concept: {question_plan.target_concept}",
            f"Image role: {role}",
            f"Objective: {objective}",
            f"Image context / content to depict: {description}",
        ]

        if question_plan.learning_objective:
            prompt_lines.append(f"Learning objective: {question_plan.learning_objective}")
        if question_plan.image_description:
            prompt_lines.append(f"Additional image context: {question_plan.image_description}")

        prompt_lines.extend(
            [
                "Required output JSON schema:",
                json.dumps(schema, ensure_ascii=False),
                "Constraints:",
                *[f"- {c}" for c in constraints],
                "If any visual element might introduce ambiguity, add a short `notes_for_illustrator`.",
            ]
        )

        llm_prompt = "\n".join(prompt_lines)

        if llm:
            return llm(llm_prompt)

        return llm_prompt
    
