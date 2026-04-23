from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.generator.image_gen import ImageGenerator
from src.generator.prompt_builder import build_image_prompt
from src.generator.question_gen import LLMQuestionGenerator
from src.planner.planner import load_plan

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GenerationOrchestrator:
    def __init__(
        self,
        *,
        image_generator: ImageGenerator | None = None,
        question_generator: LLMQuestionGenerator | None = None,
    ) -> None:
        self._image_generator = image_generator or ImageGenerator()
        self._question_generator = question_generator or LLMQuestionGenerator()

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
        prompts: list[dict[str, Any]],
        plan_json: Path | None = None,
        *,
        output_path: Path | None = None,
        run_id: str | None = None,
        image_paths: list[str] | None = None,
        mock_image: bool = False,
        mock_question: bool = False,
    ) -> dict[str, Any]:
        effective_run_id = run_id or self._generate_run_id()
        image_output_dir = self._image_output_dir(effective_run_id)

        plans_by_index = {}
        if plan_json:
            plans = load_plan(plan_json)
            for i, p in enumerate(plans, start=1):
                plans_by_index[i] = p

        results = []
        for rec in prompts:
            idx = rec.get("index")
            q_prompt = rec.get("question_prompt", "")
            img_prompt = rec.get("image_prompt")

            plan = plans_by_index.get(idx)
            if plan is None:
                from src.planner.planner import QuestionPlan as _QP

                plan = _QP(
                    target_concept=rec.get("target_concept") or f"unknown_{idx}",
                    question_type=rec.get("question_type") or "multiple_choice",
                    difficulty=rec.get("difficulty") or "medium",
                    reasoning_type=rec.get("reasoning_type") or "factoid",
                    image_role=rec.get("image_role"),
                    image_description=rec.get("image_description"),
                    learning_objective=rec.get("learning_objective"),
                    metadata=rec.get("metadata", {}),
                )

            if not img_prompt and plan is not None:
                try:
                    img_prompt = build_image_prompt(plan)
                except Exception:
                    img_prompt = None

            image_url = None
            if img_prompt:
                if mock_image:
                    image_url = f"mock://image/{effective_run_id}/{idx}.png"
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
                raise RuntimeError(f"No image prompt available for index {idx}; images are required.")

            if image_url is None:
                raise RuntimeError(f"Image generation failed for index {idx}; cannot continue without an image.")

            serialized_image_ref = self._serialize_image_ref(image_url)

            if mock_question:
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
                    "image_grounded": bool(serialized_image_ref),
                    "metadata": {
                        "reasoning_type": plan.reasoning_type if plan else "factoid",
                        "run_id": effective_run_id,
                    },
                }
                results.append({"index": idx, "question": q_obj, "image_url": serialized_image_ref})
                continue

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

        payload = {
            "run_id": effective_run_id,
            "image_dir": str(Path("data") / "images" / effective_run_id),
            "results": results,
        }

        effective_output_path = output_path or self._default_output_path(effective_run_id)
        effective_output_path.parent.mkdir(parents=True, exist_ok=True)
        with effective_output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        return payload
