from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from uuid import uuid4

from src.knowledge.schema import Question
from src.planner.planner import QuestionPlan
from src.utils.llm import LLMClient


class LLMQuestionGenerator:
    """Generate quiz questions using a large language model."""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        max_retries: int = 2,
    ) -> None:
        self._llm_client = llm_client or LLMClient()
        self._max_retries = max_retries

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are an expert assessment author. Produce one accurate question in strict JSON format "
            "matching the required schema. Do not output markdown or extra text. "
            'For multiple-choice questions, always return exactly 4 non-empty options and set '
            '"correct_answer" to either the full correct option text or one of A, B, C, D.'
        )

    @staticmethod
    def _requires_image_grounding(question_plan: QuestionPlan) -> bool:
        image_role = (question_plan.image_role or "illustrative").strip().lower()
        return image_role == "reasoning"

    @staticmethod
    def _repair_prompt(
        *,
        original_prompt: str,
        invalid_payload_text: str,
        validation_error: str,
        require_image_grounding: bool,
    ) -> str:
        repair_instructions = [
            "Your previous JSON was invalid. Return corrected JSON only.",
            "Keep the same target concept and overall intent.",
            "For multiple-choice questions, return exactly 4 non-empty options.",
            'Set "correct_answer" to either one option text or A/B/C/D.',
        ]
        if require_image_grounding:
            repair_instructions.append(
                "The question must require visual evidence from the associated image and mention concrete visual cues."
            )
        repair_text = " ".join(repair_instructions)
        return (
            f"{original_prompt}\n\n"
            f"{repair_text}\n\n"
            f"Validation error: {validation_error}\n\n"
            "Previous invalid JSON:\n"
            f"{invalid_payload_text}"
        )

    @staticmethod
    def _is_image_grounded(question: Question) -> bool:
        text_blob = f"{question.question_text} {question.explanation}".lower()
        grounding_terms = {
            "image",
            "figure",
            "diagram",
            "shown",
            "visual",
            "depicted",
            "illustration",
            "label",
            "arrow",
            "orientation",
            "position",
        }
        return any(term in text_blob for term in grounding_terms)

    @staticmethod
    def _extract_json_payload(raw_text: str) -> dict[str, object]:
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM output is not valid JSON.") from exc

        if not isinstance(parsed, dict):
            raise ValueError("LLM output must be a JSON object.")
        return parsed

    @staticmethod
    def _build_question_from_payload(
        payload: dict[str, object],
        *,
        question_plan: QuestionPlan,
        image_path: str | None,
    ) -> Question:
        question_text = str(payload.get("question_text", "")).strip()
        options_raw = payload.get("options", [])
        options: list[str] = []
        if isinstance(options_raw, list):
            options = [str(item).strip() for item in options_raw if str(item).strip()]

        correct_answer = str(payload.get("correct_answer", "")).strip()
        explanation = str(payload.get("explanation", "")).strip()

        question = Question(
            id=f"q_{uuid4().hex[:10]}",
            question_text=question_text,
            options=options,
            correct_answer=correct_answer,
            explanation=explanation,
            target_concept=question_plan.target_concept,
            difficulty=question_plan.difficulty,
            question_type=question_plan.question_type,
            associated_image=image_path,
            image_grounded=LLMQuestionGenerator._requires_image_grounding(question_plan),
            metadata={"reasoning_type": question_plan.reasoning_type, **question_plan.metadata},
        )
        question.validate()
        return question

    @staticmethod
    def _default_plan() -> QuestionPlan:
        return QuestionPlan(
            target_concept="unspecified",
            question_type="multiple-choice",
            difficulty="medium",
            reasoning_type="factual",
            image_role="illustrative",
            image_description="unspecified",
            metadata={},
        )
    
    def inference(
        self,
        question_prompt: str,
        *,
        question_plan: QuestionPlan | None = None,
        image_path: str | None = None,
    ) -> Question:
        """Generate a quiz question with options, answer, explanation, and optional image grounding."""
        prompt = question_prompt.strip()
        if not prompt:
            raise ValueError("Question prompt cannot be empty.")

        # Ensure the effective plan exists; images are mandatory
        effective_plan = question_plan or self._default_plan()
        if not image_path:
            raise ValueError("Image is required by plan but no image_path provided to generator.")
        require_image_grounding = self._requires_image_grounding(effective_plan)

        last_error: Exception | None = None
        previous_output: str | None = None
        for attempt in range(self._max_retries + 1):
            attempt_prompt = prompt
            if attempt > 0 and previous_output is not None and last_error is not None:
                attempt_prompt = self._repair_prompt(
                    original_prompt=prompt,
                    invalid_payload_text=previous_output,
                    validation_error=str(last_error),
                    require_image_grounding=require_image_grounding,
                )
            elif require_image_grounding and attempt > 0:
                attempt_prompt = (
                    f"{prompt}\n\n"
                    "Retry instruction: The output must require visual evidence from the associated image. "
                    "Mention concrete visual cues in question_text or explanation. "
                    "For multiple-choice questions, return exactly 4 non-empty options."
                )

            try:
                llm_output = self._llm_client.complete(attempt_prompt, system_prompt=self._system_prompt())
                previous_output = llm_output
                payload = self._extract_json_payload(llm_output)
                question = self._build_question_from_payload(
                    payload,
                    question_plan=effective_plan,
                    image_path=image_path,
                )

                if require_image_grounding and not self._is_image_grounded(question):
                    raise ValueError("Generated question is not image-grounded.")

                return question
            except Exception as exc:
                last_error = exc

        raise RuntimeError("Failed to generate a valid question after retries.") from last_error

    @staticmethod
    def to_dict(question: Question) -> dict[str, object]:
        # Support both dataclass and Pydantic BaseModel `Question` types
        if is_dataclass(question):
            return asdict(question)

        # Pydantic v2: model_dump; v1: dict
        if hasattr(question, "model_dump"):
            return question.model_dump()

        if hasattr(question, "dict"):
            return question.dict()

        raise TypeError("Unsupported question type for serialization")
    
