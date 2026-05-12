from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.question_formats import QuestionFormatProfile


@dataclass(frozen=True)
class QuizWorkflowRequest:
    document_path: Path
    output_root: Path | None = None
    run_id: str | None = None
    num_questions: int = 5
    difficulty_distribution: dict[str, float] | None = None
    image_paths: list[str] | None = None
    mock_image: bool = False
    mock_question: bool = False
    generation_mode: str = "topic_agentic"
    question_format_profile: QuestionFormatProfile | dict[str, float] | str | None = None
    html_graph: bool = True


@dataclass(frozen=True)
class StageExecutionResult:
    name: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentProcessingResult:
    parsed_document: Any
    extracted: dict[str, Any]
    artifacts: dict[str, str]


@dataclass
class KnowledgeGraphResult:
    graph_result: Any
    artifacts: dict[str, str]


@dataclass
class QuizPlanResult:
    plans: list[Any]
    planner_name: str
    artifacts: dict[str, str]


@dataclass
class QuizGenerationResult:
    payload: dict[str, Any]
    artifacts: dict[str, str]


@dataclass
class QuizWorkflowResult:
    run_id: str
    run_root: Path
    manifest: Path
    log_path: Path
    artifacts: dict[str, str]
    stages: dict[str, str]
    document: dict[str, Any]
    extracted: dict[str, Any]
    graph: dict[str, Any]
    graph_summary: dict[str, Any]
    plans: list[dict[str, Any]]
    generation: dict[str, Any]
