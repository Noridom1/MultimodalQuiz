from __future__ import annotations

import datetime as dt
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None  # type: ignore[assignment]

from src.document_understanding.extractor import DocumentExtractor
from src.document_understanding.chunking import build_semantic_chunks, parse_markdown_blocks
from src.document_understanding.parser import parse_document
from src.generator.orchestrator import GenerationOrchestrator
from src.knowledge.kg_builder import build_knowledge_graph_workflow, export_graph_bundle
from src.planner.planner import QuizPlanner
from src.utils.io import append_jsonl, relative_path, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


DEFAULT_DIFFICULTY_DISTRIBUTION = {
    "easy": 0.4,
    "medium": 0.4,
    "hard": 0.2,
}


def _empty_extracted_payload() -> dict[str, object]:
    return {
        "concepts": [],
        "definitions": {},
        "relations": [],
        "examples": [],
    }


def _generate_run_id(document_path: Path) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{document_path.stem}_{uuid4().hex[:6]}"


@dataclass(frozen=True)
class RunContext:
    output_root: Path
    run_id: str
    run_root: Path
    document_dir: Path
    extraction_dir: Path
    graph_dir: Path
    planning_dir: Path
    generation_dir: Path
    logs_dir: Path
    manifest_path: Path
    log_path: Path

    @classmethod
    def create(
        cls,
        document_path: Path,
        *,
        output_root: str | Path | None = None,
        run_id: str | None = None,
    ) -> "RunContext":
        root = Path(output_root) if output_root is not None else PROJECT_ROOT / "outputs"
        effective_run_id = run_id or _generate_run_id(document_path)
        run_root = root / effective_run_id
        document_dir = run_root / "document"
        extraction_dir = run_root / "extraction"
        graph_dir = run_root / "graph"
        planning_dir = run_root / "planning"
        generation_dir = run_root / "generation"
        logs_dir = run_root / "logs"

        for path in (document_dir, extraction_dir, graph_dir, planning_dir, generation_dir, logs_dir):
            path.mkdir(parents=True, exist_ok=True)

        return cls(
            output_root=root,
            run_id=effective_run_id,
            run_root=run_root,
            document_dir=document_dir,
            extraction_dir=extraction_dir,
            graph_dir=graph_dir,
            planning_dir=planning_dir,
            generation_dir=generation_dir,
            logs_dir=logs_dir,
            manifest_path=run_root / "manifest.json",
            log_path=logs_dir / "pipeline.log",
        )


def _log_event(context: RunContext, stage: str, event: str, message: str, **details: object) -> None:
    append_jsonl(
        context.log_path,
        {
            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            "run_id": context.run_id,
            "stage": stage,
            "event": event,
            "message": message,
            "details": details,
        },
    )


def _rename_artifact(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return target
    if target.exists():
        target.unlink()
    source.replace(target)
    return target


@dataclass
class QuizGenerationPipeline:
    """End-to-end pipeline from document understanding through generation."""

    output_root: str | Path | None = None
    html_graph: bool = True

    def run(
        self,
        document_path: str | Path,
        *,
        output_root: str | Path | None = None,
        run_id: str | None = None,
        num_questions: int = 5,
        difficulty_distribution: dict[str, float] | None = None,
        image_paths: list[str] | None = None,
        mock_image: bool = False,
        mock_question: bool = False,
    ) -> dict[str, object]:
        document_path = Path(document_path)
        if not document_path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")

        context = RunContext.create(
            document_path,
            output_root=output_root if output_root is not None else self.output_root,
            run_id=run_id,
        )
        effective_distribution = difficulty_distribution or DEFAULT_DIFFICULTY_DISTRIBUTION

        extractor_backend = os.getenv("QUIZGEN_EXTRACTOR_BACKEND", "langchain")
        extractor_provider = os.getenv("QUIZGEN_LLM_PROVIDER", "openai")
        extractor_granularity = os.getenv("QUIZGEN_EXTRACTION_GRANULARITY", "balanced")
        extractor_model = os.getenv("QUIZGEN_LLM_MODEL")
        kg_chunk_max_tokens = int(os.getenv("QUIZGEN_KG_MAX_TOKENS", "280"))
        kg_overlap_blocks = int(os.getenv("QUIZGEN_KG_OVERLAP_BLOCKS", "1"))

        _log_event(
            context,
            "pipeline",
            "started",
            "Pipeline started",
            document_path=str(document_path),
            output_root=str(context.output_root),
        )

        stage_status: dict[str, str] = {}
        artifacts: dict[str, str] = {}
        active_stage = "pipeline"

        try:
            active_stage = "parse"
            _log_event(context, active_stage, "started", "Parsing document")
            parsed_document = parse_document(document_path)
            parsed_path = context.document_dir / "parsed_document.json"
            write_json(parsed_path, asdict(parsed_document))
            artifacts["parsed_document"] = relative_path(parsed_path, PROJECT_ROOT)
            stage_status[active_stage] = "completed"
            _log_event(
                context,
                active_stage,
                "completed",
                "Document parsed",
                sections=len(parsed_document.sections),
                paragraphs=len(parsed_document.paragraphs),
                figures=len(parsed_document.figures),
            )

            active_stage = "extract"
            _log_event(context, active_stage, "started", "Extracting semantic knowledge")
            chunk_blocks = parse_markdown_blocks(parsed_document.markdown, source_file=document_path)
            semantic_chunks = build_semantic_chunks(
                chunk_blocks,
                max_tokens=kg_chunk_max_tokens,
                overlap_blocks=kg_overlap_blocks,
            )
            extractor = DocumentExtractor(
                backend=extractor_backend,
                provider=extractor_provider,
                granularity=extractor_granularity,
                model=extractor_model,
            )
            extracted = (
                extractor.extract_chunks(semantic_chunks, source_file=str(document_path))
                if parsed_document.markdown.strip()
                else {
                    **_empty_extracted_payload(),
                    "chunk_extractions": [],
                    "summary": {"chunk_count": 0, "concept_count": 0, "relation_count": 0},
                }
            )
            extracted_path = context.extraction_dir / "extracted.json"
            write_json(extracted_path, extracted)
            artifacts["extracted"] = relative_path(extracted_path, PROJECT_ROOT)
            stage_status[active_stage] = "completed"
            _log_event(
                context,
                active_stage,
                "completed",
                "Semantic extraction finished",
                chunks=len(extracted.get("chunk_extractions", [])),
                concepts=len(extracted.get("concepts", [])),
                definitions=len(extracted.get("definitions", {})),
                relations=len(extracted.get("relations", [])),
                examples=len(extracted.get("examples", [])),
            )

            active_stage = "graph"
            _log_event(context, active_stage, "started", "Building knowledge graph")
            graph_result = build_knowledge_graph_workflow(
                {
                    "markdown": parsed_document.markdown,
                    "sections": parsed_document.sections,
                    "paragraphs": parsed_document.paragraphs,
                    "figures": parsed_document.figures,
                    "captions": parsed_document.captions,
                },
                extracted,
                source_file=document_path,
                max_tokens=kg_chunk_max_tokens,
                overlap_blocks=kg_overlap_blocks,
            )
            document_graph = graph_result.graph
            graph_export = export_graph_bundle(
                document_graph,
                output_dir=context.graph_dir,
                html=self.html_graph,
                checkpoints=graph_result.checkpoints,
            )
            graph_json_path = _rename_artifact(graph_export["graph_json"], context.graph_dir / "graph.json")
            networkx_json_path = _rename_artifact(graph_export["networkx_json"], context.graph_dir / "graph_networkx.json")
            html_path = None
            if "html" in graph_export:
                html_path = _rename_artifact(graph_export["html"], context.graph_dir / "graph.html")

            artifacts["graph"] = relative_path(graph_json_path, PROJECT_ROOT)
            artifacts["graph_networkx"] = relative_path(networkx_json_path, PROJECT_ROOT)
            if html_path is not None:
                artifacts["graph_html"] = relative_path(html_path, PROJECT_ROOT)
            for key in (
                "hierarchy",
                "chunks",
                "artifact_links",
                "extraction_raw",
                "canonicalization",
                "merge_review",
                "merge_application",
                "graph_consolidated",
                "topic_candidates",
                "topic_consolidation",
                "topics",
                "graph_validation",
            ):
                if key in graph_export:
                    artifacts[key] = relative_path(graph_export[key], PROJECT_ROOT)
            stage_status[active_stage] = "completed"
            graph_summary = document_graph.summary()
            _log_event(
                context,
                active_stage,
                "completed",
                "Knowledge graph built",
                node_count=graph_summary["node_count"],
                edge_count=graph_summary["edge_count"],
                validation_passed=graph_result.validation.passed,
            )
            if not graph_result.validation.passed:
                raise RuntimeError("Knowledge graph validation failed: " + "; ".join(graph_result.validation.errors))

            active_stage = "plan"
            _log_event(context, active_stage, "started", "Generating quiz plan")
            planner = QuizPlanner(knowledge_graph=document_graph)
            plans = planner.plan(num_questions=num_questions, difficulty_distribution=effective_distribution)
            plan_path = context.planning_dir / "quiz_plan.json"
            planner.save_plan(plans, plan_path)
            artifacts["plan"] = relative_path(plan_path, PROJECT_ROOT)
            stage_status[active_stage] = "completed"
            _log_event(
                context,
                active_stage,
                "completed",
                "Quiz plan generated",
                question_count=len(plans),
            )

            active_stage = "generate"
            _log_event(context, active_stage, "started", "Generating images and questions")
            generator = GenerationOrchestrator()
            generation_result = generator.run(
                plan_path,
                output_dir=context.generation_dir,
                artifact_root=context.run_root,
                run_id=context.run_id,
                image_paths=image_paths,
                mock_image=mock_image,
                mock_question=mock_question,
            )

            generation_artifacts = generation_result.get("artifacts", {})
            if isinstance(generation_artifacts, dict):
                for key, value in generation_artifacts.items():
                    if isinstance(value, Path):
                        artifacts[f"generation_{key}"] = relative_path(value, PROJECT_ROOT)

            questions_path = context.generation_dir / "questions.json"
            quiz_package_path = context.generation_dir / "quiz_package.json"
            image_artifacts_path = context.generation_dir / "image_artifacts.json"
            artifacts["questions"] = relative_path(questions_path, PROJECT_ROOT)
            artifacts["quiz_package"] = relative_path(quiz_package_path, PROJECT_ROOT)
            artifacts["image_artifacts"] = relative_path(image_artifacts_path, PROJECT_ROOT)
            stage_status[active_stage] = "completed"
            _log_event(
                context,
                active_stage,
                "completed",
                "Generation finished",
                question_count=len(generation_result.get("questions", [])),
                image_count=len(generation_result.get("image_artifacts", [])),
            )

            manifest = {
                "run_id": context.run_id,
                "created_at": dt.datetime.utcnow().isoformat() + "Z",
                "source_document": relative_path(document_path, PROJECT_ROOT),
                "output_root": relative_path(context.run_root, PROJECT_ROOT),
                "stages": stage_status,
                "artifacts": {
                    **artifacts,
                    "pipeline_log": relative_path(context.log_path, PROJECT_ROOT),
                },
                "config": {
                    "extractor_backend": extractor_backend,
                    "extractor_provider": extractor_provider,
                    "extractor_granularity": extractor_granularity,
                    "extractor_model": extractor_model,
                    "kg_chunk_max_tokens": kg_chunk_max_tokens,
                    "kg_overlap_blocks": kg_overlap_blocks,
                    "num_questions": num_questions,
                    "difficulty_distribution": effective_distribution,
                    "html_graph": self.html_graph,
                    "mock_image": mock_image,
                    "mock_question": mock_question,
                },
            }
            write_json(context.manifest_path, manifest)
            _log_event(context, "pipeline", "completed", "Pipeline completed", manifest=str(context.manifest_path))

            return {
                "run_id": context.run_id,
                "run_root": context.run_root,
                "manifest": context.manifest_path,
                "log_path": context.log_path,
                "artifacts": artifacts,
                "stages": stage_status,
                "document": asdict(parsed_document),
                "extracted": extracted,
                "graph": document_graph.model_dump(mode="json"),
                "graph_summary": graph_summary,
                "plans": [asdict(plan) for plan in plans],
                "generation": generation_result,
            }
        except Exception as exc:
            _log_event(context, active_stage, "failed", "Pipeline failed", error=str(exc))
            raise
