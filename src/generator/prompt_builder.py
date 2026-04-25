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
        # knowledge_context may be a string or a list of chunk dicts
        if isinstance(context, list):
            # Format multiple chunks into a readable block
            parts = []
            for i, chunk in enumerate(context, 1):
                cid = chunk.get("id") if isinstance(chunk, dict) else None
                sb = chunk.get("source_block_id") if isinstance(chunk, dict) else None
                conf = chunk.get("confidence") if isinstance(chunk, dict) else None
                text = chunk.get("text") if isinstance(chunk, dict) else str(chunk)
                header = f"[Chunk {i}] Block ID: {sb or cid} | Confidence: {conf or 'N/A'}"
                parts.append(f"{header}\n{text}\n")
            context_str = "\n---\n".join(parts)
        else:
            context_str = str(context or "")

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
        if context_str.strip():
            prompt_lines.append("Knowledge context:")
            # Indent the context for readability
            for line in context_str.splitlines():
                prompt_lines.append(f"  {line}")

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
        """Ask an LLM to produce a concise, human-readable image-generation instruction.

        The LLM is asked to return a short natural-language prompt (1-3 sentences) suitable
        for an image generation model or an illustrator. Do NOT return JSON or code fences.

        If `llm` is provided it will be called with the composed prompt and its raw text
        response will be returned. Otherwise the composed LLM prompt string is returned
        for an external caller to send to an LLM.
        """
        role = (question_plan.image_role or "illustration").strip()
        description = question_plan.image_description or question_plan.target_concept
        objective = question_plan.learning_objective or "support quiz question answering"
        prompt_lines = [
            "SYSTEM: You are an expert image-generation prompt engineer. Produce a short, natural-language instruction",
            "that an image model or human illustrator can follow. Do NOT return JSON or code fences.",
            "Return 1-3 concise sentences describing composition, labels to include, color guidance, and any accessibility alt text.",
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

        prompt_lines.append(
            "Constraints: Keep it concise (1-3 sentences). Use clear labels, high contrast, and avoid copyrighted or trademarked elements. Include a one-line alt text sentence at the end prefixed with 'Alt:'."
        )

        llm_prompt = "\n".join(prompt_lines)

        if llm:
            return llm(llm_prompt)

        return llm_prompt
    
