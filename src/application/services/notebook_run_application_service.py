from __future__ import annotations

import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any, Callable

from api.config import Settings
from api.repository import NotebookRepository
from src.application.services.notebook_quiz_application_service import NotebookQuizApplicationService
from src.utils.llm import LLMClient

logger = logging.getLogger(__name__)


class NotebookRunApplicationService:
    """Application-layer workflow for executing and finalizing notebook quiz runs."""

    def __init__(
        self,
        *,
        settings: Settings,
        repo: NotebookRepository,
        quiz_workflows: NotebookQuizApplicationService,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.quiz_workflows = quiz_workflows
        self._llm_client = llm_client or LLMClient()

    def generate_quiz(
        self,
        *,
        notebook_id: str,
        notebook: dict[str, Any],
        source: dict[str, Any],
        num_questions: int,
        mock_question: bool,
        question_format_distribution: dict[str, float] | None,
        hydrate_run: Callable[[dict[str, Any] | None], dict[str, Any] | None],
        now_token: str,
        slugify: Callable[[str], str],
    ) -> dict[str, Any]:
        document_path = Path(source["local_path"])
        pending_run = self.repo.create_run(
            {
                "notebook_id": notebook_id,
                "source_id": source["id"],
                "run_id": f"pending-{slugify(notebook['title'])}-{now_token}",
                "title": f"{source['title']} quiz",
                "status": "running",
                "num_questions": num_questions,
                "manifest": {},
                "summary": {},
                "artifact_paths": {},
            }
        )

        self.repo.create_message(
            notebook_id,
            "assistant",
            f"Generating a {num_questions}-question multimodal quiz from `{source['title']}`.",
            kind="system",
        )

        try:
            effective_mock_image = True
            logger.info(
                "Notebook quiz generation requested notebook_id=%s source_id=%s num_questions=%s mock_question=%s",
                notebook_id,
                source["id"],
                num_questions,
                mock_question,
            )
            result = self.quiz_workflows.generate_for_source(
                document_path=document_path,
                num_questions=num_questions,
                mock_image=effective_mock_image,
                mock_question=mock_question,
                question_format_profile=question_format_distribution,
            )
            run_id = result["run_id"]
            manifest_path = Path(result["manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary = self._build_run_summary(
                notebook_id,
                result,
                source_title=str(source.get("title") or ""),
                notebook_title=str(notebook.get("title") or ""),
            )
            updated = self.repo.update_run(
                pending_run["run_id"],
                {
                    "run_id": run_id,
                    "title": str(summary.get("title") or ""),
                    "status": "completed",
                    "manifest": manifest,
                    "summary": summary,
                    "artifact_paths": summary["artifact_paths"],
                },
            )
            if summary.get("hero_image"):
                self.repo.update_notebook(notebook_id, {"cover_image": summary["hero_image"]})
            self.repo.create_message(
                notebook_id,
                "assistant",
                f"Finished `{summary['title']}` with {summary['question_count']} questions across "
                f"{len(summary['concepts'])} core concepts.",
            )
            hydrated = hydrate_run(updated or self.repo.get_run_by_run_id(run_id))
            if not hydrated:
                raise RuntimeError("Run hydration failed after quiz generation")
            return hydrated
        except Exception:
            logger.exception(
                "Notebook quiz generation failed notebook_id=%s source_id=%s",
                notebook_id,
                source.get("id"),
            )
            self.repo.update_run(pending_run["run_id"], {"status": "failed", "summary": {"error": "generation failed"}})
            raise

    def _build_run_summary(
        self,
        notebook_id: str,
        result: dict[str, Any],
        *,
        source_title: str,
        notebook_title: str,
    ) -> dict[str, Any]:
        run_root = Path(result["run_root"])
        quiz_package_path = run_root / "generation" / "quiz_package.json"
        quiz_package = json.loads(quiz_package_path.read_text(encoding="utf-8"))
        published = self._publish_run_artifacts(notebook_id, result["run_id"], run_root, quiz_package)
        generated_title = self._generate_quiz_title(
            quiz_package,
            source_title=source_title,
            notebook_title=notebook_title,
        )
        first_image = next(
            (item.get("image_url", "") for item in quiz_package.get("results", []) if item.get("image_url")),
            "",
        )
        hero_image = published.get(first_image) or (self._artifact_url(run_root / first_image) if first_image else "")
        concepts = list(
            dict.fromkeys(
                item["question"].get("target_concept", "")
                for item in quiz_package.get("results", [])
                if item.get("question", {}).get("target_concept")
            )
        )
        return {
            "title": generated_title,
            "question_count": len(quiz_package.get("results", [])),
            "hero_image": hero_image,
            "concepts": concepts,
            "results": [
                {
                    "index": item["index"],
                    "question_text": item["question"]["question_text"],
                    "question_type": item["question"].get("question_type", "multiple_choice"),
                    "difficulty": item["question"]["difficulty"],
                    "target_concept": item["question"]["target_concept"],
                    "explanation": item["question"]["explanation"],
                    "correct_answer": item["question"]["correct_answer"],
                    "options": item["question"].get("options", []),
                    "metadata": self._client_question_metadata(item["question"]),
                    "image_url": published.get(item.get("image_url", ""))
                    or (self._artifact_url(run_root / item["image_url"]) if item.get("image_url") else ""),
                }
                for item in quiz_package.get("results", [])
            ],
            "artifact_paths": {
                "manifest": published.get("manifest") or self._artifact_url(Path(result["manifest"])),
                "quiz_package": published.get("quiz_package") or self._artifact_url(quiz_package_path),
                "graph": published.get("graph") or self._artifact_url(run_root / "graph" / "graph.json"),
                "graph_html": published.get("graph_html") or self._artifact_url(run_root / "graph" / "graph.html"),
            },
        }

    def _client_question_metadata(self, question: dict[str, Any]) -> dict[str, Any]:
        metadata = question.get("metadata")
        if not isinstance(metadata, dict):
            return {}
        keys = {"matching_left", "matching_right", "matching_solution", "matching_pairs"}
        return {k: metadata[k] for k in keys if k in metadata}

    def _generate_quiz_title(
        self,
        quiz_package: dict[str, Any],
        *,
        source_title: str,
        notebook_title: str,
    ) -> str:
        fallback = self._quiz_title_fallback(source_title=source_title, notebook_title=notebook_title)
        results = quiz_package.get("results")
        if not isinstance(results, list) or not results:
            return fallback

        concepts = list(
            dict.fromkeys(
                str(item.get("question", {}).get("target_concept", "")).strip()
                for item in results
                if isinstance(item, dict)
            )
        )
        concepts = [concept for concept in concepts if concept][:6]
        question_stems = []
        for item in results[:6]:
            if not isinstance(item, dict):
                continue
            question = item.get("question") or {}
            if not isinstance(question, dict):
                continue
            text = str(question.get("question_text", "")).strip()
            if text:
                question_stems.append(text)

        prompt_lines = [
            "Create one concise and descriptive quiz title.",
            "Requirements:",
            "- Return plain text only (no markdown, no quotes).",
            "- Between 2 and 6 words.",
            "- Avoid generic words like Run, Quiz Set, or Test 1.",
            "- Capture the topic and learning intent.",
            f"Source title: {source_title or 'Unknown source'}",
            f"Notebook title: {notebook_title or 'Unknown notebook'}",
            f"Key concepts: {', '.join(concepts) if concepts else 'N/A'}",
            "Question stems:",
            *[f"- {stem}" for stem in question_stems],
        ]

        try:
            raw = self._llm_client.complete(
                "\n".join(prompt_lines),
                system_prompt=(
                    "You create short educational quiz titles. "
                    "Respond with one title only."
                ),
            )
        except Exception:
            return fallback
        sanitized = self._sanitize_generated_title(raw)
        return sanitized or fallback

    def _sanitize_generated_title(self, title: str) -> str:
        cleaned = str(title or "").strip()
        if not cleaned:
            return ""
        cleaned = cleaned.strip("`\"' ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"[.!?;:,]+$", "", cleaned).strip()
        words = [part for part in cleaned.split(" ") if part]
        if len(words) > 6:
            cleaned = " ".join(words[:6])
        if len(cleaned) > 120:
            cleaned = cleaned[:120].rstrip()
        return cleaned

    def _quiz_title_fallback(self, *, source_title: str, notebook_title: str) -> str:
        source = str(source_title or "").strip()
        if source:
            return self._sanitize_generated_title(f"{source} Quiz") or "Untitled Quiz"
        notebook = str(notebook_title or "").strip()
        if notebook:
            return self._sanitize_generated_title(f"{notebook} Quiz") or "Untitled Quiz"
        return "Untitled Quiz"

    def _publish_run_artifacts(
        self,
        notebook_id: str,
        run_id: str,
        run_root: Path,
        quiz_package: dict[str, Any],
    ) -> dict[str, str]:
        published: dict[str, str] = {}
        manifest_path = run_root / "manifest.json"
        graph_path = run_root / "graph" / "graph.json"
        graph_html_path = run_root / "graph" / "graph.html"
        quiz_package_path = run_root / "generation" / "quiz_package.json"

        for key, path, content_type in (
            ("manifest", manifest_path, "application/json"),
            ("quiz_package", quiz_package_path, "application/json"),
            ("graph", graph_path, "application/json"),
            ("graph_html", graph_html_path, "text/html"),
        ):
            if path.exists():
                try:
                    published[key] = self.repo.upload_blob(
                        f"notebooks/{notebook_id}/runs/{run_id}/{path.name}",
                        path.read_bytes(),
                        content_type,
                    )
                except Exception:
                    continue

        for item in quiz_package.get("results", []):
            image_ref = item.get("image_url", "")
            if not image_ref:
                continue
            image_path = run_root / image_ref
            if not image_path.exists():
                continue
            content_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
            try:
                published[image_ref] = self.repo.upload_blob(
                    f"notebooks/{notebook_id}/runs/{run_id}/images/{image_path.name}",
                    image_path.read_bytes(),
                    content_type,
                )
            except Exception:
                continue
        return published

    def _artifact_url(self, path: Path) -> str:
        relative = path.resolve().relative_to(self.settings.project_root.resolve()).as_posix()
        return f"/api/artifacts/{relative}"
