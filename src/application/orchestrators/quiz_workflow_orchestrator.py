from __future__ import annotations

import datetime as dt
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None  # type: ignore[assignment]

from src.application.contracts.workflow import QuizWorkflowRequest, QuizWorkflowResult
from src.application.services.artifact_service import RunArtifactService
from src.application.services.document_processing_service import DocumentProcessingService
from src.application.services.knowledge_graph_service import KnowledgeGraphService
from src.application.services.quiz_generation_service import QuizGenerationService
from src.application.services.quiz_planning_service import QuizPlanningService
from src.application.services.run_context import RunContext
from src.application.services.run_logging import RunLogService
from src.question_formats import (
    QuestionFormatProfile,
    coerce_format_profile,
    load_format_profile_from_pipeline_config,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_DIFFICULTY_DISTRIBUTION = {
    "easy": 0.4,
    "medium": 0.4,
    "hard": 0.2,
}


def resolve_question_format_profile(
    explicit: QuestionFormatProfile | dict[str, float] | str | None,
    project_root: Path,
) -> QuestionFormatProfile:
    if explicit is not None:
        return coerce_format_profile(explicit)
    cfg_path = project_root / "configs" / "default.yaml"
    if cfg_path.exists():
        try:
            import yaml

            with open(cfg_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            pipe = data.get("pipeline") or {}
            loaded = load_format_profile_from_pipeline_config(pipe)
            if loaded is not None:
                return loaded
        except Exception:
            logger.exception("Failed to load question_format_distribution from default.yaml")
    return coerce_format_profile(None)


class QuizWorkflowOrchestrator:
    def __init__(
        self,
        *,
        project_root: Path | None = None,
        html_graph: bool = True,
        artifact_service: RunArtifactService | None = None,
        log_service: RunLogService | None = None,
        document_service: DocumentProcessingService | None = None,
        knowledge_graph_service: KnowledgeGraphService | None = None,
        planning_service: QuizPlanningService | None = None,
        generation_service: QuizGenerationService | None = None,
    ) -> None:
        self._project_root = Path(project_root) if project_root is not None else PROJECT_ROOT
        self._artifact_service = artifact_service or RunArtifactService(project_root=self._project_root)
        self._log_service = log_service or RunLogService()
        self._document_service = document_service or DocumentProcessingService(
            artifact_service=self._artifact_service
        )
        self._knowledge_graph_service = knowledge_graph_service or KnowledgeGraphService(
            artifact_service=self._artifact_service,
            html_graph=html_graph,
        )
        self._planning_service = planning_service or QuizPlanningService(
            artifact_service=self._artifact_service
        )
        self._generation_service = generation_service or QuizGenerationService(
            artifact_service=self._artifact_service
        )
        self._html_graph = html_graph

    def run(self, request: QuizWorkflowRequest) -> QuizWorkflowResult:
        if request.generation_mode not in ("topic_agentic", "legacy"):
            raise ValueError(
                f"Invalid generation_mode: {request.generation_mode}. "
                "Must be 'topic_agentic' or 'legacy'."
            )
        if request.generation_mode == "legacy":
            logger.warning(
                "Using legacy generation mode 'legacy'. This mode is deprecated and will be removed "
                "in a future version. Please use generation_mode='topic_agentic' (default)."
            )

        document_path = Path(request.document_path)
        if not document_path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")

        context = RunContext.create(
            document_path,
            project_root=self._project_root,
            output_root=request.output_root,
            run_id=request.run_id,
        )
        effective_distribution = request.difficulty_distribution or DEFAULT_DIFFICULTY_DISTRIBUTION
        resolved_format_profile = resolve_question_format_profile(
            request.question_format_profile,
            self._project_root,
        )

        extractor_backend = os.getenv("QUIZGEN_EXTRACTOR_BACKEND", "langchain")
        extractor_provider = os.getenv("QUIZGEN_LLM_PROVIDER", "openai")
        extractor_granularity = os.getenv("QUIZGEN_EXTRACTION_GRANULARITY", "balanced")
        extractor_model = os.getenv("QUIZGEN_LLM_MODEL")
        kg_chunk_max_tokens = int(os.getenv("QUIZGEN_KG_MAX_TOKENS", "280"))
        kg_overlap_blocks = int(os.getenv("QUIZGEN_KG_OVERLAP_BLOCKS", "1"))

        self._log_service.log_event(
            context,
            "pipeline",
            "started",
            "Pipeline started",
            document_path=str(document_path),
            output_root=str(context.output_root),
            question_format_distribution=dict(resolved_format_profile.distribution),
        )

        stage_status: dict[str, str] = {}
        artifacts: dict[str, str] = {}
        active_stage = "pipeline"

        try:
            active_stage = "parse"
            self._log_service.log_event(context, active_stage, "started", "Parsing document")
            processed = self._document_service.process(
                context=context,
                document_path=document_path,
                extractor_backend=extractor_backend,
                extractor_provider=extractor_provider,
                extractor_granularity=extractor_granularity,
                extractor_model=extractor_model,
                kg_chunk_max_tokens=kg_chunk_max_tokens,
                kg_overlap_blocks=kg_overlap_blocks,
            )
            artifacts.update(processed.artifacts)
            parsed_document = processed.parsed_document
            extracted = processed.extracted
            stage_status[active_stage] = "completed"
            self._log_service.log_event(
                context,
                active_stage,
                "completed",
                "Document parsed and semantic extraction finished",
                sections=len(parsed_document.sections),
                paragraphs=len(parsed_document.paragraphs),
                figures=len(parsed_document.figures),
                chunks=len(extracted.get("chunk_extractions", [])),
                concepts=len(extracted.get("concepts", [])),
                definitions=len(extracted.get("definitions", {})),
                relations=len(extracted.get("relations", [])),
                examples=len(extracted.get("examples", [])),
            )

            active_stage = "graph"
            self._log_service.log_event(context, active_stage, "started", "Building knowledge graph")
            graph_result = self._knowledge_graph_service.build(
                context=context,
                document_path=document_path,
                processed=processed,
                kg_chunk_max_tokens=kg_chunk_max_tokens,
                kg_overlap_blocks=kg_overlap_blocks,
            )
            artifacts.update(graph_result.artifacts)
            document_graph = graph_result.graph_result.graph
            graph_summary = document_graph.summary()
            stage_status[active_stage] = "completed"
            self._log_service.log_event(
                context,
                active_stage,
                "completed",
                "Knowledge graph built",
                node_count=graph_summary["node_count"],
                edge_count=graph_summary["edge_count"],
                validation_passed=graph_result.graph_result.validation.passed,
            )
            if not graph_result.graph_result.validation.passed:
                raise RuntimeError(
                    "Knowledge graph validation failed: "
                    + "; ".join(graph_result.graph_result.validation.errors)
                )

            active_stage = "plan"
            self._log_service.log_event(context, active_stage, "started", "Generating quiz plan")
            plan_result = self._planning_service.plan(
                context=context,
                graph_result=graph_result,
                generation_mode=request.generation_mode,
                num_questions=request.num_questions,
                difficulty_distribution=effective_distribution,
                format_profile=resolved_format_profile,
            )
            artifacts.update(plan_result.artifacts)
            plans = plan_result.plans
            stage_status[active_stage] = "completed"
            self._log_service.log_event(
                context,
                active_stage,
                "completed",
                "Quiz plan generated",
                question_count=len(plans),
                generation_mode=request.generation_mode,
                planner_name=plan_result.planner_name,
                question_format_distribution=dict(resolved_format_profile.distribution),
            )

            active_stage = "generate"
            self._log_service.log_event(context, active_stage, "started", "Generating images and questions")
            generation_result = self._generation_service.generate(
                context=context,
                image_paths=request.image_paths,
                mock_image=request.mock_image,
                mock_question=request.mock_question,
            )
            artifacts.update(generation_result.artifacts)
            generation_payload = generation_result.payload
            stage_status[active_stage] = "completed"
            self._log_service.log_event(
                context,
                active_stage,
                "completed",
                "Generation finished",
                question_count=len(generation_payload.get("questions", [])),
                image_count=len(generation_payload.get("image_artifacts", [])),
            )

            manifest = {
                "run_id": context.run_id,
                "created_at": dt.datetime.utcnow().isoformat() + "Z",
                "source_document": self._artifact_service.project_relative(document_path),
                "output_root": self._artifact_service.project_relative(context.run_root),
                "stages": stage_status,
                "artifacts": {
                    **artifacts,
                    "pipeline_log": self._artifact_service.project_relative(context.log_path),
                },
                "config": {
                    "extractor_backend": extractor_backend,
                    "extractor_provider": extractor_provider,
                    "extractor_granularity": extractor_granularity,
                    "extractor_model": extractor_model,
                    "kg_chunk_max_tokens": kg_chunk_max_tokens,
                    "kg_overlap_blocks": kg_overlap_blocks,
                    "num_questions": request.num_questions,
                    "difficulty_distribution": effective_distribution,
                    "html_graph": self._html_graph,
                    "mock_image": request.mock_image,
                    "mock_question": request.mock_question,
                    "question_format_distribution": dict(resolved_format_profile.distribution),
                },
            }
            self._artifact_service.write_manifest(context, manifest)
            self._log_service.log_event(
                context,
                "pipeline",
                "completed",
                "Pipeline completed",
                manifest=str(context.manifest_path),
            )

            return QuizWorkflowResult(
                run_id=context.run_id,
                run_root=context.run_root,
                manifest=context.manifest_path,
                log_path=context.log_path,
                artifacts=artifacts,
                stages=stage_status,
                document=asdict(parsed_document),
                extracted=extracted,
                graph=document_graph.model_dump(mode="json"),
                graph_summary=graph_summary,
                plans=[asdict(plan) for plan in plans],
                generation=generation_payload,
            )
        except Exception as exc:
            self._log_service.log_event(context, active_stage, "failed", "Pipeline failed", error=str(exc))
            raise
