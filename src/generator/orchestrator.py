from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.generator.image_gen import ImageGenerator
from src.generator.prompt_builder import PromptBuilder
from src.generator.question_gen import LLMQuestionGenerator
from src.planner.planner import QuestionPlan, load_plan
from src.utils.io import relative_path, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    def _default_output_dir(run_id: str) -> Path:
        return PROJECT_ROOT / "data" / "questions" / run_id

    @staticmethod
    def _image_output_dir(run_id: str) -> Path:
        return PROJECT_ROOT / "data" / "images" / run_id

    @staticmethod
    def _safe_text(value: object | None) -> str:
        return "" if value is None else str(value).strip()

    @staticmethod
    def load_prompts_from_dir(prompts_dir: Path) -> list[dict[str, Any]]:
        """Load question_prompt.json and image_prompts.json from a directory and merge them into records."""
        q_path = prompts_dir / "question_prompt.json"
        i_path = prompts_dir / "image_prompts.json"
        questions: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []
        if q_path.exists():
            questions = json.loads(q_path.read_text(encoding="utf-8"))
        if i_path.exists():
            images = json.loads(i_path.read_text(encoding="utf-8"))

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
        return records

    def build_prompts_from_plan(self, plan_json: Path) -> list[dict[str, Any]]:
        """Convert a saved quiz plan JSON into prompt records for generation."""
        plans = load_plan(plan_json)
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
        return records

    @staticmethod
    def _serialize_image_ref(image_value: str | None, *, artifact_root: Path | None = None) -> str | None:
        if not image_value:
            return image_value

        lowered = image_value.lower()
        if lowered.startswith(("http://", "https://", "data:", "mock://")):
            return image_value

        if artifact_root is not None:
            return relative_path(Path(image_value), artifact_root)

        return Path(image_value).name

    def run(
        self,
        plan_json: Path,
        *,
        prompts: list[dict[str, Any]] | None = None,
        output_path: Path | None = None,
        output_dir: Path | None = None,
        artifact_root: Path | None = None,
        run_id: str | None = None,
        image_paths: list[str] | None = None,
        mock_image: bool = False,
        mock_question: bool = False,
    ) -> dict[str, Any]:
        effective_run_id = run_id or self._generate_run_id()
        effective_output_dir = Path(output_dir) if output_dir is not None else None
        if effective_output_dir is None and output_path is not None:
            effective_output_dir = Path(output_path).parent
        if effective_output_dir is None:
            effective_output_dir = self._default_output_dir(effective_run_id)
        effective_output_dir.mkdir(parents=True, exist_ok=True)

        effective_artifact_root = Path(artifact_root) if artifact_root is not None else effective_output_dir
        image_output_dir = effective_output_dir / "images"
        image_output_dir.mkdir(parents=True, exist_ok=True)

        plan_records = self.build_prompts_from_plan(plan_json)
        effective_prompts = prompts or plan_records
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

        question_records: list[dict[str, Any]] = []
        image_artifacts: list[dict[str, Any]] = []
        questions: list[dict[str, Any]] = []
        for rec in effective_prompts:
            idx = rec.get("index")
            q_prompt = rec.get("question_prompt", "")
            img_prompt = rec.get("image_prompt")

            plan = plans_by_index.get(idx)
            if plan is None:
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
                try:
                    q_prompt = self._prompt_builder.build_question_prompt(plan)
                except Exception:
                    q_prompt = ""

            if not img_prompt and plan is not None:
                try:
                    img_prompt = self._prompt_builder.build_image_prompt(plan)
                except Exception:
                    img_prompt = None

            image_url = None
            image_status = "skipped"
            if img_prompt:
                if mock_image:
                    image_url = f"mock://image/{effective_run_id}/{idx}.png"
                    image_status = "generated"
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
                        image_status = "generated"
            else:
                raise RuntimeError(f"No image prompt available for index {idx}; images are required.")

            if image_url is None:
                raise RuntimeError(f"Image generation failed for index {idx}; cannot continue without an image.")

            serialized_image_ref = self._serialize_image_ref(image_url, artifact_root=effective_artifact_root)
            image_local_path = None
            if serialized_image_ref and not serialized_image_ref.startswith(("http://", "https://", "data:", "mock://")):
                image_local_path = relative_path(Path(image_url), effective_artifact_root)

            image_artifacts.append(
                {
                    "index": idx,
                    "status": image_status,
                    "image_prompt": img_prompt,
                    "source_url": image_url,
                    "local_path": image_local_path,
                    "image_ref": serialized_image_ref,
                }
            )

            if mock_question:
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
                question_records.append(
                    {
                        "index": idx,
                        "question": q_obj,
                        "image_url": serialized_image_ref,
                        "question_prompt": q_prompt,
                        "image_prompt": img_prompt,
                    }
                )
                questions.append(q_obj)
                continue

            if not q_prompt:
                raise RuntimeError(f"No question prompt available for index {idx}; cannot generate question.")

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
            question_records.append(
                {
                    "index": idx,
                    "question": question_dict,
                    "image_url": serialized_image_ref,
                    "question_prompt": q_prompt,
                    "image_prompt": img_prompt,
                }
            )
            questions.append(question_dict)

        payload = {
            "run_id": effective_run_id,
            "image_dir": relative_path(image_output_dir, effective_artifact_root),
            "results": question_records,
        }

        questions_path = effective_output_dir / "questions.json"
        quiz_package_path = effective_output_dir / "quiz_package.json"
        image_artifacts_path = effective_output_dir / "image_artifacts.json"

        write_json(questions_path, questions)
        write_json(quiz_package_path, payload)
        write_json(image_artifacts_path, image_artifacts)

        if output_path is not None:
            legacy_output_path = Path(output_path)
            if legacy_output_path.suffix:
                write_json(legacy_output_path, payload)
            else:
                legacy_output_path.mkdir(parents=True, exist_ok=True)
                write_json(legacy_output_path / "quiz_package.json", payload)

        return {
            "run_id": effective_run_id,
            "questions": questions,
            "question_records": question_records,
            "image_artifacts": image_artifacts,
            "artifacts": {
                "questions": questions_path,
                "quiz_package": quiz_package_path,
                "image_artifacts": image_artifacts_path,
                "image_dir": image_output_dir,
            },
            "payload": payload,
        }
