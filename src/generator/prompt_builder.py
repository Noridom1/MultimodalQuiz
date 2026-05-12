from __future__ import annotations

import json
from typing import Callable, Optional, Dict, Any

from src.planner.planner import QuestionPlan
from src.question_formats import normalize_question_type

class PromptBuilder:
    """Build image and question prompts from a question plan."""

    @staticmethod
    def _schema_and_constraints(
        question_type: str,
        *,
        image_rule: str,
    ) -> tuple[dict[str, object], list[str]]:
        qt = normalize_question_type(question_type)
        base_constraints = [
            "Return ONLY valid JSON with no markdown fences.",
            "Keep explanation concise and instructional (1-3 sentences).",
            image_rule,
        ]

        if qt == "multiple_choice":
            schema: dict[str, object] = {
                "question_text": "string",
                "options": ["string", "string", "string", "string"],
                "correct_answer": "string",
                "explanation": "string",
            }
            constraints = base_constraints + [
                "Include exactly the keys shown in the schema.",
                "Provide exactly 4 non-empty options.",
                'Set correct_answer to the full correct option text or A/B/C/D.',
            ]
            return schema, constraints

        if qt == "true_false":
            schema = {
                "question_text": "string",
                "options": ["True", "False"],
                "correct_answer": "string",
                "explanation": "string",
            }
            constraints = base_constraints + [
                "Include exactly the keys shown in the schema.",
                "options must be exactly [\"True\", \"False\"] in that order.",
                'correct_answer must be \"True\" or \"False\" (or match the option text).',
                "Phrase question_text as a clear declarative statement to evaluate.",
            ]
            return schema, constraints

        if qt == "fill_in_blank":
            schema = {
                "question_text": "string",
                "options": [],
                "correct_answer": "string",
                "explanation": "string",
            }
            constraints = base_constraints + [
                "Include exactly the keys shown in the schema.",
                "question_text must contain ____ or {blank} where the learner fills in.",
                "options must be an empty JSON array [].",
                "correct_answer is the short text that belongs in the blank (one phrase).",
            ]
            return schema, constraints

        if qt == "matching":
            schema = {
                "question_text": "string",
                "options": [],
                "correct_answer": "string",
                "explanation": "string",
                "matching_left": ["string"],
                "matching_right": ["string"],
                "matching_solution": [[0, 0]],
            }
            constraints = base_constraints + [
                "Include exactly the keys shown in the schema.",
                "matching_left and matching_right must be the same length (>= 2), same order as displayed.",
                "matching_solution is a list of [left_index, right_index] pairs forming a perfect one-to-one match.",
                "options must be an empty JSON array [].",
                "correct_answer must briefly summarize the correct pairings in plain language.",
            ]
            return schema, constraints

        schema = {
            "question_text": "string",
            "options": ["string"],
            "correct_answer": "string",
            "explanation": "string",
        }
        constraints = base_constraints + [
            "Include the keys in the schema; adapt options length to the question style.",
        ]
        return schema, constraints

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

        schema, constraints = PromptBuilder._schema_and_constraints(
            question_plan.question_type,
            image_rule=image_rule,
        )

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
        mp = (question_plan.metadata or {}).get("matching_pairs")
        if normalize_question_type(question_plan.question_type) == "matching" and isinstance(mp, list) and mp:
            prompt_lines.append(f"Planner draft pairs (refine, do not contradict): {json.dumps(mp, ensure_ascii=True)}")

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
        metadata = getattr(question_plan, "metadata", {}) or {}
        # If detailed prompt requested or no-spoiler required, emit full structured prompt
        if metadata.get("detailed_image_prompt") or metadata.get("no_spoiler"):
            return PromptBuilder.build_image_prompt_detailed(question_plan)

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
    def build_image_prompt_detailed(question_plan: QuestionPlan) -> str:
        """Produce a multi-section, detailed image prompt suitable for image models.

        The output is a short natural-language instruction but explicitly names
        sections such as Intent, Scope, Style, Labels, No-Spoiler rules and an Alt: line.
        """
        metadata = getattr(question_plan, "metadata", {}) or {}
        audience = _safe_meta(metadata, "audience", "general learners")
        style = _safe_meta(metadata, "style", "clean vector (flat colors)")
        year = _safe_meta(metadata, "year", "")
        place_list = metadata.get("place_list") or []
        palette = _safe_meta(metadata, "palette", "colorblind-safe palette")
        resolution = _safe_meta(metadata, "resolution", "3000×2000 px")
        no_spoiler = bool(metadata.get("no_spoiler", False))

        places = ", ".join(place_list) if place_list else question_plan.image_description or question_plan.target_concept

        parts: list[str] = []
        parts.append(f"Intent: Educational map for {audience} to illustrate {question_plan.target_concept}.")
        if year:
            parts.append(f"Scope: circa {year}, show {places}.")
        else:
            parts.append(f"Scope: show {places}.")

        parts.append(f"Style: {style}; palette: {palette}; resolution: {resolution}.")
        parts.append(
            "Visual hierarchy: draw invasion routes as thick, semi-transparent red arrows with arrowheads indicating direction; "
            "shade occupied areas in a muted color, use neutral gray for borders, and keep region labels larger than city/event labels."
        )

        parts.append(
            "Labels & typography: prefer sans-serif labels for place names, serif for title; keep labels concise and avoid overlapping; "
            "include a legend, scale bar (e.g., 50 km), and north arrow."
        )

        if no_spoiler:
            parts.append(
                "No-Spoiler: DO NOT render any individual personal or leader names or explicit answer text. "
                "Replace such labels with placeholders like 'Leader A' or numbered markers (1,2,3). "
                "Do not include nameplates, captions, or other text that reveals quiz answers."
            )
        else:
            parts.append("No-Spoiler: not requested for this prompt.")

        parts.append(
            "Annotations: mark major events with numbered icons (crossed swords for battles, scrolls for treaties) and concise labels (max 12 words)."
        )

        parts.append("Negative: no modern infrastructure, no logos, no photographs, no copyrighted imagery.")

        alt_text = _safe_meta(metadata, "alt", f"Historical map of {places}; red arrows indicate invasion routes; numbered markers indicate key events.")
        parts.append(f"Alt: {alt_text}")

        # Join into a compact natural-language instruction (1-4 sentences) but retain clarity.
        return " ".join(parts)

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
            "Constraints: Keep it concise (1-3 sentences). Use clear labels, high contrast, and avoid copyrighted or trademarked elements. "
            "IMPORTANT: Do NOT render any text that directly reveals the quiz answer; replace person/place names with placeholders like 'Leader A' or 'Site 1'. "
            "Include a one-line alt text sentence at the end prefixed with 'Alt:'."
        )

        llm_prompt = "\n".join(prompt_lines)

        if llm:
            return llm(llm_prompt)

        return llm_prompt
    
