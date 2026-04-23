from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.generator.image_gen import ImageGenerator
from src.generator.prompt_builder import PromptBuilder
from src.generator.question_gen import LLMQuestionGenerator
from src.planner.planner import QuestionPlan, load_plan

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


class GenerationOrchestrator:
    def __init__(
        self,
        *,
        image_generator: ImageGenerator | None = None,
        question_generator: LLMQuestionGenerator | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self._image_generator = image_generator or ImageGenerator()
        self._question_generator = question_generator or LLMQuestionGenerator()
        self._prompt_builder = prompt_builder or PromptBuilder()

    @staticmethod
    def _generate_run_id() -> str:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"qset_{timestamp}_{uuid4().hex[:6]}"

    @staticmethod
    def _default_output_path(run_id: str) -> Path:
        return PROJECT_ROOT / "data" / "questions" / f"{run_id}.json"

    @staticmethod
    def _image_output_dir(run_id: str) -> Path:
        return PROJECT_ROOT / "data" / "images" / run_id

    @staticmethod
    def load_prompts_from_dir(prompts_dir: Path) -> list[dict[str, Any]]:
        """Load question_prompt.json and image_prompts.json from a directory and merge them into records."""
        LOGGER.info("Loading prompts from directory: %s", prompts_dir)
        q_path = prompts_dir / "question_prompt.json"
        i_path = prompts_dir / "image_prompts.json"
        questions: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []
        if q_path.exists():
            questions = json.loads(q_path.read_text(encoding="utf-8"))
        if i_path.exists():
            images = json.loads(i_path.read_text(encoding="utf-8"))
        LOGGER.info(
            "Loaded prompt source files: questions=%d image_prompts=%d",
            len(questions),
            len(images),
        )

        images_by_index = {item["index"]: item for item in images}
        records: list[dict[str, Any]] = []
        for q in questions:
            idx = q.get("index")
            rec = {
                "index": idx,
                "target_concept": q.get("target_concept"),
                "question_prompt": q.get("question_prompt"),
                "question_type": q.get("question_type"),
                "difficulty": q.get("difficulty"),
                "reasoning_type": q.get("reasoning_type"),
                "image_role": q.get("image_role"),
                "image_description": q.get("image_description"),
                "learning_objective": q.get("learning_objective"),
                "metadata": q.get("metadata", {}),
                "image_prompt": None,
            }
            img = images_by_index.get(idx)
            if img:
                rec["image_prompt"] = img.get("image_prompt")
            records.append(rec)
        LOGGER.info("Prepared %d merged prompt records from %s", len(records), prompts_dir)
        return records

    def build_prompts_from_plan(self, plan_json: Path) -> list[dict[str, Any]]:
        """Convert a saved quiz plan JSON into prompt records for generation."""
        LOGGER.info("Building prompts from plan: %s", plan_json)
        plans = load_plan(plan_json)
        LOGGER.info("Loaded %d question plans", len(plans))
        records: list[dict[str, Any]] = []
        for idx, plan in enumerate(plans, start=1):
            records.append(
                {
                    "index": idx,
                    "target_concept": plan.target_concept,
                    "question_prompt": self._prompt_builder.build_question_prompt(plan),
                    "question_type": plan.question_type,
                    "difficulty": plan.difficulty,
                    "reasoning_type": plan.reasoning_type,
                    "image_role": plan.image_role,
                    "image_description": plan.image_description,
                    "learning_objective": plan.learning_objective,
                    "metadata": plan.metadata,
                    "image_prompt": self._prompt_builder.build_image_prompt(plan),
                }
            )
        LOGGER.info("Built %d prompt records from plan %s", len(records), plan_json)
        return records

    @staticmethod
    def _serialize_image_ref(image_value: str | None) -> str | None:
        if not image_value:
            return image_value

        lowered = image_value.lower()
        if lowered.startswith(("http://", "https://", "data:", "mock://")):
            return image_value

        return Path(image_value).name

    def run(
        self,
        plan_json: Path,
        *,
        prompts: list[dict[str, Any]] | None = None,
        output_path: Path | None = None,
        run_id: str | None = None,
        image_paths: list[str] | None = None,
        mock_image: bool = False,
        mock_question: bool = False,
    ) -> dict[str, Any]:
        effective_run_id = run_id or self._generate_run_id()
        image_output_dir = self._image_output_dir(effective_run_id)
        LOGGER.info(
            "Starting generation run: run_id=%s plan=%s mock_image=%s mock_question=%s",
            effective_run_id,
            plan_json,
            mock_image,
            mock_question,
        )
        LOGGER.info("Image outputs will be written under %s", image_output_dir)

        plan_records = self.build_prompts_from_plan(plan_json)
        effective_prompts = prompts or plan_records
        prompt_source = "provided prompts override" if prompts is not None else "plan-derived prompts"
        LOGGER.info(
            "Using %d prompt records from %s (plan_records=%d)",
            len(effective_prompts),
            prompt_source,
            len(plan_records),
        )
        plans_by_index = {
            record["index"]: QuestionPlan(
                target_concept=record.get("target_concept") or f"unknown_{record['index']}",
                question_type=record.get("question_type") or "multiple_choice",
                difficulty=record.get("difficulty") or "medium",
                reasoning_type=record.get("reasoning_type") or "factoid",
                image_role=record.get("image_role"),
                image_description=record.get("image_description"),
                learning_objective=record.get("learning_objective"),
                metadata=record.get("metadata", {}),
            )
            for record in plan_records
        }

        results = []
        total_records = len(effective_prompts)
        for position, rec in enumerate(effective_prompts, start=1):
            idx = rec.get("index")
            q_prompt = rec.get("question_prompt", "")
            img_prompt = rec.get("image_prompt")
            LOGGER.info(
                "Processing record %d/%d (index=%s, target_concept=%s)",
                position,
                total_records,
                idx,
                rec.get("target_concept"),
            )

            plan = plans_by_index.get(idx)
            if plan is None:
                LOGGER.warning("No plan record found for index=%s; building fallback plan from prompt record", idx)
                plan = QuestionPlan(
                    target_concept=rec.get("target_concept") or f"unknown_{idx}",
                    question_type=rec.get("question_type") or "multiple_choice",
                    difficulty=rec.get("difficulty") or "medium",
                    reasoning_type=rec.get("reasoning_type") or "factoid",
                    image_role=rec.get("image_role"),
                    image_description=rec.get("image_description"),
                    learning_objective=rec.get("learning_objective"),
                    metadata=rec.get("metadata", {}),
                )

            if not q_prompt and plan is not None:
                LOGGER.info("Question prompt missing for index=%s; rebuilding from plan", idx)
                try:
                    q_prompt = self._prompt_builder.build_question_prompt(plan)
                except Exception:
                    LOGGER.exception("Failed to rebuild question prompt for index=%s", idx)
                    q_prompt = ""

            if not img_prompt and plan is not None:
                LOGGER.info("Image prompt missing for index=%s; rebuilding from plan", idx)
                try:
                    img_prompt = self._prompt_builder.build_image_prompt(plan)
                except Exception:
                    LOGGER.exception("Failed to rebuild image prompt for index=%s", idx)
                    img_prompt = None

            image_url = None
            if img_prompt:
                LOGGER.info("Generating image for index=%s", idx)
                if mock_image:
                    image_url = f"mock://image/{effective_run_id}/{idx}.png"
                    LOGGER.info("Using mock image output for index=%s: %s", idx, image_url)
                else:
                    image_url = self._image_generator.generate(
                        img_prompt,
                        image_paths=image_paths,
                        output_dir=image_output_dir,
                        file_stem=str(idx),
                    )
                    if not image_url:
                        image_url = None
                    else:
                        LOGGER.info("Image generation completed for index=%s: %s", idx, image_url)
            else:
                LOGGER.error("No image prompt available for index=%s", idx)
                raise RuntimeError(f"No image prompt available for index {idx}; images are required.")

            if image_url is None:
                LOGGER.error("Image generation failed for index=%s", idx)
                raise RuntimeError(f"Image generation failed for index {idx}; cannot continue without an image.")

            serialized_image_ref = self._serialize_image_ref(image_url)
            LOGGER.info("Serialized image reference for index=%s: %s", idx, serialized_image_ref)

            if mock_question:
                LOGGER.info("Using mock question output for index=%s", idx)
                image_grounded = bool(plan and (plan.image_role or "").strip().lower() == "reasoning")
                q_obj = {
                    "id": f"q_mock_{idx}",
                    "question_text": f"Mock question for {plan.target_concept if plan else idx}",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": "A",
                    "explanation": "This is a mock explanation.",
                    "target_concept": plan.target_concept if plan else f"unknown_{idx}",
                    "difficulty": plan.difficulty if plan else "medium",
                    "question_type": plan.question_type if plan else "multiple_choice",
                    "associated_image": serialized_image_ref,
                    "image_grounded": image_grounded,
                    "metadata": {
                        "reasoning_type": plan.reasoning_type if plan else "factoid",
                        "run_id": effective_run_id,
                    },
                }
                results.append({"index": idx, "question": q_obj, "image_url": serialized_image_ref})
                LOGGER.info("Completed record index=%s with mock question output", idx)
                continue

            if not q_prompt:
                LOGGER.error("No question prompt available for index=%s", idx)
                raise RuntimeError(f"No question prompt available for index {idx}; cannot generate question.")

            LOGGER.info("Generating question for index=%s", idx)
            question = self._question_generator.inference(
                q_prompt,
                question_plan=plan,
                image_path=image_url,
            )
            question_dict = LLMQuestionGenerator.to_dict(question)
            question_dict["associated_image"] = serialized_image_ref
            metadata = question_dict.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["run_id"] = effective_run_id
            question_dict["metadata"] = metadata
            results.append({"index": idx, "question": question_dict, "image_url": serialized_image_ref})
            LOGGER.info(
                "Completed record index=%s question_id=%s",
                idx,
                question_dict.get("id"),
            )

        payload = {
            "run_id": effective_run_id,
            "image_dir": str(Path("data") / "images" / effective_run_id),
            "results": results,
        }

        effective_output_path = output_path or self._default_output_path(effective_run_id)
        effective_output_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Writing generation payload to %s", effective_output_path)
        with effective_output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        LOGGER.info(
            "Generation run completed: run_id=%s records=%d output=%s",
            effective_run_id,
            len(results),
            effective_output_path,
        )
        return payload
