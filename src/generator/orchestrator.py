from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4
import concurrent.futures
import math

from src.generator.image_gen import ImageGenerator
from src.generator.prompt_builder import PromptBuilder
from src.generator.question_gen import LLMQuestionGenerator
from src.planner.planner import QuestionPlan, load_plan
from src.utils.io import relative_path, write_json
from src.utils.llm import LLMClient

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
        # LLM client: required for image prompt construction (always used)
        self._llm_client = LLMClient()

    def run_with_topic_planner(
        self,
        graph,
        *,
        total_questions: int,
        output_dir: Path | None = None,
        run_id: str | None = None,
        mock_image: bool = False,
        mock_question: bool = False,
        difficulty_distribution: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Generate quiz questions using the topic-agentic planner.
        
        Args:
            graph: MultimodalDocumentGraph to generate from.
            total_questions: Total number of questions to generate.
            output_dir: Optional output directory; defaults to {PROJECT_ROOT}/data/questions/{run_id}.
            run_id: Optional run ID; auto-generated if not provided.
            mock_image: If True, skip image generation (use mocked paths).
            mock_question: If True, skip question generation (use placeholder questions).
            difficulty_distribution: Optional dict mapping difficulty to proportion.
            
        Returns:
            Generation result dict with questions, metadata, paths, etc.
        """
        # Import here to avoid circular imports
        from src.planner.topic_planner import TopicAgenticPlanner
        from src.knowledge.schema import MultimodalDocumentGraph
        import tempfile
        
        # Validate graph
        if not isinstance(graph, MultimodalDocumentGraph):
            raise TypeError("graph must be a MultimodalDocumentGraph instance")
        
        # Create planner and generate plan
        planner = TopicAgenticPlanner(knowledge_graph=graph)
        plans = planner.plan(
            total_questions=total_questions,
            difficulty_distribution=difficulty_distribution,
        )
        
        # Save plan to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            plan_data = {"questions": []}
            for idx, plan in enumerate(plans):
                plan_dict = {
                    "index": idx,
                    "target_concept": plan.target_concept,
                    "question_type": plan.question_type,
                    "difficulty": plan.difficulty,
                    "reasoning_type": plan.reasoning_type,
                    "image_role": plan.image_role,
                    "image_description": plan.image_description,
                    "learning_objective": plan.learning_objective,
                    "tested_fact_block_id": plan.tested_fact_block_id,
                    "metadata": plan.metadata,
                }
                plan_data["questions"].append(plan_dict)
            json.dump(plan_data, f)
            plan_path = Path(f.name)
        
        try:
            # Run generation with the plan
            effective_run_id = run_id or self._generate_run_id()
            result = self.run(
                plan_path,
                output_dir=output_dir,
                run_id=effective_run_id,
                mock_image=mock_image,
                mock_question=mock_question,
            )
            return result
        finally:
            # Clean up temporary plan file
            try:
                plan_path.unlink()
            except:
                pass

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
            # Always use the LLM to generate a plain-language image prompt (no JSON)
            raw_llm_response = self._prompt_builder.build_image_prompt_via_llm(
                plan, llm=self._llm_client.complete
            )
            final_image_prompt = str(raw_llm_response).strip() if raw_llm_response is not None else ""
            if not final_image_prompt:
                raise RuntimeError(f"LLM returned empty image prompt for plan index {idx}")
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
                    "tested_fact_block_id": plan.tested_fact_block_id,
                    "metadata": plan.metadata,
                    "image_prompt": final_image_prompt,
                    "raw_image_llm_response": raw_llm_response,
                }
            )
        LOGGER.info("Built %d prompt records from plan %s", len(records), plan_json)
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
        # Directory to store question prompt logs (prompt text + attached knowledge_context)
        prompt_logs_dir = effective_output_dir / "prompt_logs"
        prompt_logs_dir.mkdir(parents=True, exist_ok=True)
        run_started_at = time.perf_counter()

        plan_records = self.build_prompts_from_plan(plan_json)
        effective_prompts = prompts or plan_records
        prompt_source = "provided prompts override" if prompts is not None else "plan-derived prompts"
        LOGGER.info(
            "Starting generation run_id=%s plan=%s output_dir=%s artifact_root=%s mock_image=%s mock_question=%s image_refs=%d",
            effective_run_id,
            plan_json,
            effective_output_dir,
            effective_artifact_root,
            mock_image,
            mock_question,
            len(image_paths or []),
        )
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
                tested_fact_block_id=record.get("tested_fact_block_id"),
                metadata=record.get("metadata", {}),
            )
            for record in plan_records
        }

        question_records: list[dict[str, Any]] = []
        image_artifacts: list[dict[str, Any]] = []
        questions: list[dict[str, Any]] = []
        total_records = len(effective_prompts)

        # Parallel image generation: submit image tasks concurrently to reduce total runtime.
        image_futures: dict[concurrent.futures.Future, dict[str, Any]] = {}
        image_results_by_index: dict[int, dict[str, Any]] = {}
        # Determine worker count (bounded to reasonable limits)
        default_workers = min(8, (len(effective_prompts) or 1))
        image_workers = default_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=image_workers) as executor:
            for rec in effective_prompts:
                idx = rec.get("index")
                img_prompt = rec.get("image_prompt")
                # Skip submission if no image prompt
                if not img_prompt:
                    continue

                def _submit_image(idx_local: int, prompt_local: str):
                    try:
                        if mock_image:
                            image_url = f"mock://image/{effective_run_id}/{idx_local}.png"
                            return {"index": idx_local, "status": "generated", "image_url": image_url}
                        else:
                            image_url = self._image_generator.generate(
                                prompt_local,
                                image_paths=image_paths,
                                output_dir=image_output_dir,
                                file_stem=str(idx_local),
                            )
                            if not image_url:
                                return {"index": idx_local, "status": "failed", "image_url": None}
                            return {"index": idx_local, "status": "generated", "image_url": image_url}
                    except Exception as e:
                        LOGGER.exception("Image generation exception index=%s: %s", idx_local, e)
                        return {"index": idx_local, "status": "failed", "image_url": None}

                fut = executor.submit(_submit_image, idx, img_prompt)
                image_futures[fut] = rec

            # Collect image results as they complete
            for fut in concurrent.futures.as_completed(list(image_futures.keys())):
                rec = image_futures.get(fut)
                try:
                    res = fut.result()
                except Exception as exc:
                    LOGGER.exception("Unexpected image future failure for record %s: %s", rec.get("index"), exc)
                    res = {"index": rec.get("index"), "status": "failed", "image_url": None}
                image_results_by_index[int(res.get("index"))] = res

        # After all image tasks finished, proceed record-by-record to generate questions
        for position, rec in enumerate(effective_prompts, start=1):
            idx = rec.get("index")
            q_prompt = rec.get("question_prompt", "")
            img_prompt = rec.get("image_prompt")
            record_started_at = time.perf_counter()
            LOGGER.info(
                "Processing generation record %d/%d index=%s target_concept=%s difficulty=%s question_type=%s",
                position,
                total_records,
                idx,
                rec.get("target_concept"),
                rec.get("difficulty"),
                rec.get("question_type"),
            )
            try:
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

                LOGGER.info(
                    "Prepared prompts for index=%s question_prompt=%s image_prompt=%s",
                    idx,
                    "yes" if bool(self._safe_text(q_prompt)) else "no",
                    "yes" if bool(self._safe_text(img_prompt)) else "no",
                )

                image_url = None
                image_status = "skipped"
                if img_prompt:
                    # Retrieve parallel image generation result
                    img_res = image_results_by_index.get(int(idx), {})
                    image_url = img_res.get("image_url")
                    image_status = img_res.get("status", "failed")
                    LOGGER.info(
                        "Image generation result for index=%s status=%s image_ref=%s",
                        idx,
                        image_status,
                        image_url,
                    )
                else:
                    raise RuntimeError(f"No image prompt available for index {idx}; images are required.")

                if image_url is None:
                    LOGGER.error("Image generation failed for index=%s", idx)
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
                    LOGGER.info(
                        "Completed record index=%s with mock question id=%s elapsed=%.2fs",
                        idx,
                        q_obj["id"],
                        time.perf_counter() - record_started_at,
                    )
                    continue

                if not q_prompt:
                    LOGGER.error("No question prompt available for index=%s", idx)
                    raise RuntimeError(f"No question prompt available for index {idx}; cannot generate question.")

                LOGGER.info("Starting question generation for index=%s", idx)
                # Persist the prompt and attached knowledge context for offline inspection
                try:
                    # plan may be None if fallback was used earlier
                    plan_for_log = plan
                    knowledge_context = None
                    tested_fact = None
                    image_description = None
                    if plan_for_log is not None:
                        tested_fact = getattr(plan_for_log, "tested_fact_block_id", None)
                        image_description = getattr(plan_for_log, "image_description", None)
                        metadata = getattr(plan_for_log, "metadata", {}) or {}
                        knowledge_context = metadata.get("knowledge_context") if isinstance(metadata, dict) else None

                    log_path = prompt_logs_dir / f"question_{idx}.txt"
                    with open(log_path, "w", encoding="utf-8") as lf:
                        lf.write(f"Index: {idx}\n")
                        lf.write(f"Target concept: {plan_for_log.target_concept if plan_for_log else rec.get('target_concept')}\n")
                        lf.write(f"Tested fact block id: {tested_fact}\n")
                        lf.write(f"Image description: {image_description}\n\n")
                        lf.write("--- Attached knowledge_context (if any) ---\n")
                        lf.write((knowledge_context or "(none)") + "\n\n")
                        lf.write("--- Question prompt sent to LLM ---\n")
                        lf.write(q_prompt + "\n\n")
                        lf.write("--- Image prompt ---\n")
                        lf.write((img_prompt or "(none)") + "\n\n")
                        # If we have a raw LLM response for the image prompt, log it as well
                        raw_llm = rec.get("raw_image_llm_response")
                        if raw_llm:
                            lf.write("--- Raw LLM image-prompt response ---\n")
                            lf.write(str(raw_llm) + "\n")
                except Exception:
                    LOGGER.exception("Failed to write prompt log for index=%s", idx)
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
                LOGGER.info(
                    "Completed question generation for index=%s question_id=%s elapsed=%.2fs",
                    idx,
                    question_dict.get("id"),
                    time.perf_counter() - record_started_at,
                )
            except Exception:
                LOGGER.exception(
                    "Generation record failed %d/%d index=%s target_concept=%s",
                    position,
                    total_records,
                    idx,
                    rec.get("target_concept"),
                )
                raise

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

        LOGGER.info(
            "Completed generation run_id=%s records=%d questions=%d images=%d output_dir=%s elapsed=%.2fs",
            effective_run_id,
            total_records,
            len(questions),
            len(image_artifacts),
            effective_output_dir,
            time.perf_counter() - run_started_at,
        )

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
