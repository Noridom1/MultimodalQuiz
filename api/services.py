from __future__ import annotations

import csv
import datetime as dt
import io
import json
from collections import defaultdict
import mimetypes
import re
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from fastapi import HTTPException, UploadFile

from .auth import AuthUser
from .config import Settings
from .repository import NotebookRepository
from src.application.orchestrators.quiz_workflow_orchestrator import QuizWorkflowOrchestrator
from src.application.services.notebook_quiz_application_service import NotebookQuizApplicationService
from src.application.services.notebook_run_application_service import NotebookRunApplicationService
from src.utils.llm import LLMClient
import logging


logger = logging.getLogger(__name__)


EXPORT_FORMATS = {"zip", "pdf"}
EXPORT_DATA_FORMATS = {"json", "csv"}


class NotebookService:
    def __init__(self, settings: Settings, repo: NotebookRepository) -> None:
        self.settings = settings
        self.repo = repo
        self.quiz_workflows = NotebookQuizApplicationService(
            orchestrator=QuizWorkflowOrchestrator(
                project_root=settings.project_root,
                html_graph=True,
            ),
            output_root=settings.output_root,
        )
        self._llm_client = LLMClient()
        self.run_workflows = NotebookRunApplicationService(
            settings=settings,
            repo=repo,
            quiz_workflows=self.quiz_workflows,
            llm_client=self._llm_client,
        )

    def _require_notebook_owner(self, notebook: dict[str, Any], auth: AuthUser | None) -> None:
        if auth is None:
            return
        if notebook.get("owner_id") != auth.id:
            raise HTTPException(status_code=404, detail="Notebook not found")

    def list_notebook_cards(self, *, auth: AuthUser | None, query: str | None = None) -> list[dict[str, Any]]:
        owner = auth.id if auth else None
        notebooks = self.repo.list_notebooks(owner_id=owner, query=query)
        if not notebooks:
            return []

        ids = [str(n["id"]) for n in notebooks if n.get("id")]
        sources_all = self.repo.list_sources_for_notebook_ids(ids)
        runs_all = self.repo.list_runs_for_notebook_ids(ids)

        sources_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sources_all:
            bid = str(row.get("notebook_id") or "")
            if bid:
                sources_by[bid].append(row)

        runs_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in runs_all:
            bid = str(row.get("notebook_id") or "")
            if bid:
                runs_by[bid].append(row)
        for bid in runs_by:
            runs_by[bid].sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)

        cards: list[dict[str, Any]] = []
        for notebook in notebooks:
            self._require_notebook_owner(notebook, auth)
            nid = str(notebook.get("id") or "")
            nb_runs = runs_by.get(nid, [])
            latest = nb_runs[0] if nb_runs else None
            summary = self._run_summary_dict(latest) if latest else {}
            question_count = self._question_count_from_parsed_summary(summary)
            if not question_count and latest:
                raw_n = latest.get("num_questions")
                if isinstance(raw_n, int) and raw_n >= 0:
                    question_count = raw_n
            cover = str(notebook.get("cover_image") or "").strip()
            hero = cover or str(summary.get("hero_image") or "").strip()
            cards.append(
                {
                    **notebook,
                    "source_count": len(sources_by.get(nid, [])),
                    "run_count": len(nb_runs),
                    "question_count": question_count,
                    "hero_image": hero,
                }
            )
        return cards

    @staticmethod
    def _run_summary_dict(run: dict[str, Any] | None) -> dict[str, Any]:
        if not run:
            return {}
        summary = run.get("summary") or {}
        if isinstance(summary, str):
            try:
                summary = json.loads(summary)
            except json.JSONDecodeError:
                return {}
        return summary if isinstance(summary, dict) else {}

    @staticmethod
    def _question_count_from_parsed_summary(summary: dict[str, Any]) -> int:
        qc = summary.get("question_count")
        if isinstance(qc, int) and qc >= 0:
            return qc
        results = summary.get("results") or []
        if isinstance(results, list):
            return len(results)
        return 0

    def get_workspace(self, notebook_id: str, *, auth: AuthUser | None) -> dict[str, Any]:
        notebook = self.repo.get_notebook(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
        self._require_notebook_owner(notebook, auth)

        self.repo.update_notebook(notebook_id, {"last_opened_at": self._now()})
        sources = self.repo.list_sources(notebook_id)
        messages = self.repo.list_messages(notebook_id)
        runs = [self._hydrate_run(run) for run in self.repo.list_runs(notebook_id)]
        latest_run = runs[0] if runs else None

        if not messages:
            self.repo.create_message(
                notebook_id,
                "assistant",
                "Upload a source on the left, then generate a multimodal quiz from the Studio panel.",
                kind="system",
            )
            messages = self.repo.list_messages(notebook_id)

        return {
            "notebook": notebook,
            "sources": sources,
            "messages": messages,
            "runs": runs,
            "latest_run": latest_run,
        }

    async def add_source(
        self,
        notebook_id: str,
        *,
        upload: UploadFile,
        title: str | None,
        auth: AuthUser | None,
    ) -> dict[str, Any]:
        notebook = self.repo.get_notebook(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
        self._require_notebook_owner(notebook, auth)

        file_bytes = await upload.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        safe_name = self._slugify(upload.filename or "source")
        local_dir = self.settings.upload_root / notebook_id
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / safe_name
        local_path.write_bytes(file_bytes)

        storage_path = f"notebooks/{notebook_id}/sources/{safe_name}"
        content_type = upload.content_type or mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
        public_url = self.repo.upload_blob(storage_path, file_bytes, content_type)

        source = self.repo.create_source(
            {
                "notebook_id": notebook_id,
                "kind": "file",
                "title": title or Path(safe_name).stem.replace("-", " ").title(),
                "filename": upload.filename or safe_name,
                "content_type": content_type,
                "size_bytes": len(file_bytes),
                "storage_path": storage_path,
                "public_url": public_url,
                "local_path": str(local_path),
            }
        )
        self.repo.create_message(
            notebook_id,
            "assistant",
            f"Added source `{source['title']}`. You can generate a quiz from it now.",
            kind="system",
        )
        return source

    def add_message(self, notebook_id: str, content: str, *, auth: AuthUser | None) -> dict[str, Any]:
        notebook = self.repo.get_notebook(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
        self._require_notebook_owner(notebook, auth)

        user_message = self.repo.create_message(notebook_id, "user", content.strip())
        runs = [self._hydrate_run(run) for run in self.repo.list_runs(notebook_id)]
        sources = self.repo.list_sources(notebook_id)
        assistant_reply = self._build_assistant_reply(content, sources=sources, runs=runs)
        assistant_message = self.repo.create_message(notebook_id, "assistant", assistant_reply)
        return {"user": user_message, "assistant": assistant_message}

    def generate_quiz(
        self,
        notebook_id: str,
        *,
        source_id: str,
        num_questions: int,
        mock_image: bool,
        mock_question: bool,
        question_format_distribution: dict[str, float] | None,
        auth: AuthUser | None,
    ) -> dict[str, Any]:
        notebook = self.repo.get_notebook(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
        self._require_notebook_owner(notebook, auth)

        source = next((item for item in self.repo.list_sources(notebook_id) if item["id"] == source_id), None)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        document_path = Path(source["local_path"])
        if not document_path.exists():
            raise HTTPException(status_code=400, detail="Source file is no longer available locally")

        try:
            return self.run_workflows.generate_quiz(
                notebook_id=notebook_id,
                notebook=notebook,
                source=source,
                num_questions=num_questions,
                mock_question=mock_question,
                question_format_distribution=question_format_distribution,
                hydrate_run=self._hydrate_run,
                now_token=self._now_token(),
                slugify=self._slugify,
            )
        except Exception as exc:
            logger.exception(
                "API generate_quiz failed notebook_id=%s source_id=%s",
                notebook_id,
                source_id,
            )
            self.repo.create_message(
                notebook_id,
                "assistant",
                f"Quiz generation failed: {exc}",
                kind="system",
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    def patch_notebook(self, notebook_id: str, *, title: str | None, auth: AuthUser | None) -> dict[str, Any]:
        notebook = self.repo.get_notebook(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
        self._require_notebook_owner(notebook, auth)
        patch: dict[str, Any] = {}
        if title is not None:
            cleaned = title.strip()
            if not cleaned:
                raise HTTPException(status_code=400, detail="Title cannot be empty")
            patch["title"] = cleaned
        if not patch:
            raise HTTPException(status_code=400, detail="No updates provided")
        updated = self.repo.update_notebook(notebook_id, patch)
        if not updated:
            raise HTTPException(status_code=404, detail="Notebook not found")
        return updated

    def rename_run(self, notebook_id: str, run_id: str, title: str, *, auth: AuthUser | None) -> dict[str, Any]:
        run = self.repo.get_run_by_run_id(run_id)
        if not run or run.get("notebook_id") != notebook_id:
            raise HTTPException(status_code=404, detail="Run not found")
        notebook = self.repo.get_notebook(notebook_id)
        if notebook:
            self._require_notebook_owner(notebook, auth)
        cleaned = title.strip()
        if not cleaned:
            raise HTTPException(status_code=400, detail="Title cannot be empty")
        summary = run.get("summary") or {}
        if isinstance(summary, str):
            try:
                summary = json.loads(summary)
            except json.JSONDecodeError:
                summary = {}
        if not isinstance(summary, dict):
            summary = {}
        summary = {**summary, "title": cleaned}
        updated = self.repo.update_run(run_id, {"title": cleaned, "summary": summary})
        if not updated:
            raise HTTPException(status_code=404, detail="Run not found")
        hydrated = self._hydrate_run(updated)
        if not hydrated:
            raise HTTPException(status_code=404, detail="Run not found")
        return hydrated

    def list_quiz_list_cards(self, *, auth: AuthUser | None) -> list[dict[str, Any]]:
        owner = auth.id if auth else None
        lists = self.repo.list_quiz_lists(owner_id=owner)
        cards: list[dict[str, Any]] = []
        for row in lists:
            items = self.repo.list_quiz_list_items(row["id"])
            question_total = 0
            for item in items:
                run = self.repo.get_run_by_run_id(item["run_id"])
                if run:
                    question_total += self._question_count_from_run(run)
            cards.append(
                {
                    **row,
                    "item_count": len(items),
                    "question_count": question_total,
                }
            )
        return cards

    def create_quiz_list(self, title: str, folder_color: str | None, *, auth: AuthUser | None) -> dict[str, Any]:
        cleaned = title.strip()
        if not cleaned:
            raise HTTPException(status_code=400, detail="Title cannot be empty")
        color = self._normalize_quiz_list_folder_color(folder_color, cleaned)
        owner_id = auth.id if auth else None
        return self.repo.create_quiz_list(cleaned, color, owner_id=owner_id)

    def get_quiz_list_detail(self, list_id: str, *, auth: AuthUser | None) -> dict[str, Any]:
        row = self.repo.get_quiz_list(list_id)
        if not row:
            raise HTTPException(status_code=404, detail="List not found")
        self._require_quiz_list_owner(row, auth)
        items = self.repo.list_quiz_list_items(list_id)
        enriched: list[dict[str, Any]] = []
        question_total = 0
        for item in items:
            run = self.repo.get_run_by_run_id(item["run_id"])
            hydrated = self._hydrate_run(run) if run else None
            q_count = self._question_count_from_run(run) if run else 0
            question_total += q_count
            run_title = ""
            if hydrated:
                summary = hydrated.get("summary") or {}
                run_title = str(summary.get("title") or hydrated.get("title") or "Untitled quiz").strip()
            enriched.append(
                {
                    **item,
                    "run_title": run_title or "Untitled quiz",
                    "question_count": q_count,
                }
            )
        return {
            "list": row,
            "items": enriched,
            "item_count": len(items),
            "question_count": question_total,
        }

    def add_quiz_list_item(
        self,
        list_id: str,
        notebook_id: str,
        run_id: str,
        *,
        auth: AuthUser | None,
    ) -> dict[str, Any]:
        lst = self.repo.get_quiz_list(list_id)
        if not lst:
            raise HTTPException(status_code=404, detail="List not found")
        self._require_quiz_list_owner(lst, auth)

        if self.repo.find_quiz_list_item(list_id, run_id):
            raise HTTPException(status_code=409, detail="Quiz already in this list")

        notebook = self.repo.get_notebook(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
        self._require_notebook_owner(notebook, auth)

        run = self.repo.get_run_by_run_id(run_id)
        if not run or str(run.get("notebook_id")) != str(notebook_id):
            raise HTTPException(status_code=404, detail="Run not found")
        if run.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Only completed quizzes can be saved to a list")

        existing = self.repo.list_quiz_list_items(list_id)
        sort_order = (max((item.get("sort_order") or 0) for item in existing) + 1) if existing else 0
        created = self.repo.create_quiz_list_item(
            {
                "list_id": list_id,
                "notebook_id": notebook_id,
                "run_id": run_id,
                "sort_order": sort_order,
            }
        )
        self.repo.update_quiz_list(list_id, {})
        return created

    def remove_quiz_list_item(self, list_id: str, item_id: str, *, auth: AuthUser | None) -> None:
        lst = self.repo.get_quiz_list(list_id)
        if not lst:
            raise HTTPException(status_code=404, detail="List not found")
        self._require_quiz_list_owner(lst, auth)
        item = self.repo.get_quiz_list_item(item_id)
        if not item or str(item.get("list_id")) != str(list_id):
            raise HTTPException(status_code=404, detail="Item not found")
        self.repo.delete_quiz_list_item(item_id)
        self.repo.update_quiz_list(list_id, {})

    def _require_quiz_list_owner(self, row: dict[str, Any], auth: AuthUser | None) -> None:
        if auth is None:
            return
        if row.get("owner_id") != auth.id:
            raise HTTPException(status_code=404, detail="List not found")

    def _normalize_quiz_list_folder_color(self, color: str | None, title: str) -> str:
        allowed = ("#f0bf57", "#e37a46", "#e74c3c", "#9d6bff", "#5f6fff", "#4fb38a")
        lowered = {c.lower(): c for c in allowed}
        if color:
            stripped = color.strip()
            key = stripped.lower()
            if key in lowered:
                return lowered[key]
        palette = ["#f0bf57", "#e37a46", "#9d6bff", "#5f6fff", "#4fb38a"]
        return palette[sum(ord(ch) for ch in title) % len(palette)]

    def _question_count_from_run(self, run: dict[str, Any]) -> int:
        hydrated = self._hydrate_run(run)
        if not hydrated:
            return 0
        summary = hydrated.get("summary") or {}
        qc = summary.get("question_count")
        if isinstance(qc, int) and qc >= 0:
            return qc
        results = summary.get("results") or []
        if isinstance(results, list):
            return len(results)
        raw = hydrated.get("num_questions")
        return int(raw) if isinstance(raw, int) else 0

    def delete_notebook_run(self, notebook_id: str, run_id: str, *, auth: AuthUser | None) -> None:
        run = self.repo.get_run_by_run_id(run_id)
        if not run or run.get("notebook_id") != notebook_id:
            raise HTTPException(status_code=404, detail="Run not found")
        notebook = self.repo.get_notebook(notebook_id)
        if notebook:
            self._require_notebook_owner(notebook, auth)
        self.repo.delete_run(run_id)

    def export_run(
        self,
        notebook_id: str,
        run_id: str,
        *,
        auth: AuthUser | None,
        export_format: str = "zip",
        data_format: str = "json",
    ) -> tuple[bytes, str, str]:
        fmt = (export_format or "zip").lower()
        if fmt not in EXPORT_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported export format '{export_format}'")
        data_fmt = (data_format or "json").lower()
        if data_fmt not in EXPORT_DATA_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported data format '{data_format}'")

        run, summary, results = self._load_exportable_run(notebook_id, run_id, auth=auth)
        run_root = self._run_root_from_manifest(run.get("manifest") or {}, run_id)
        title = str(summary.get("title") or run.get("title") or "Untitled quiz")
        base_name = self._slugify(title) or self._slugify(run_id)

        if fmt == "pdf":
            payload = self._build_run_pdf(run, summary, results, run_root, title)
            return payload, f"{base_name}.pdf", "application/pdf"

        payload = self._build_run_zip(run, summary, results, run_root, title, run_id, data_fmt)
        return payload, f"{base_name}.zip", "application/zip"

    def _load_exportable_run(
        self,
        notebook_id: str,
        run_id: str,
        *,
        auth: AuthUser | None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        run = self._hydrate_run(self.repo.get_run_by_run_id(run_id))
        if not run or run.get("notebook_id") != notebook_id:
            raise HTTPException(status_code=404, detail="Run not found")
        notebook = self.repo.get_notebook(notebook_id)
        if notebook:
            self._require_notebook_owner(notebook, auth)
        if run.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Only completed quizzes can be exported")

        summary = run.get("summary") or {}
        results = summary.get("results") or []
        if not isinstance(results, list) or not results:
            raise HTTPException(status_code=400, detail="Run has no quiz questions to export")
        return run, summary, results

    def _build_run_zip(
        self,
        run: dict[str, Any],
        summary: dict[str, Any],
        results: list[dict[str, Any]],
        run_root: Path,
        title: str,
        run_id: str,
        data_format: str,
    ) -> bytes:
        questions: list[dict[str, Any]] = []
        zip_buffer = io.BytesIO()
        used_image_names: set[str] = set()

        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for ordinal, item in enumerate(results, start=1):
                if not isinstance(item, dict):
                    continue
                image_entry = ""
                image_path = self._resolve_export_image_path(item.get("image_url", ""), run_root)
                if image_path:
                    image_entry = self._export_image_name(ordinal, image_path.name, used_image_names)
                    archive.write(image_path, f"quiz-export/{image_entry}")

                questions.append(
                    {
                        "index": item.get("index", ordinal),
                        "question_text": item.get("question_text", ""),
                        "options": item.get("options", []) if isinstance(item.get("options"), list) else [],
                        "correct_answer": item.get("correct_answer", ""),
                        "explanation": item.get("explanation", ""),
                        "difficulty": item.get("difficulty", ""),
                        "target_concept": item.get("target_concept", ""),
                        "image": image_entry,
                    }
                )

            concepts = summary.get("concepts", []) if isinstance(summary.get("concepts"), list) else []

            if data_format == "csv":
                archive.writestr(
                    "quiz-export/quiz.csv",
                    self._questions_to_csv(questions),
                )
                meta = {
                    "schema_version": 1,
                    "run_id": run.get("run_id", run_id),
                    "title": title,
                    "question_count": len(questions),
                    "concepts": concepts,
                    "questions_file": "quiz.csv",
                }
                archive.writestr(
                    "quiz-export/manifest.json",
                    json.dumps(meta, indent=2, ensure_ascii=False),
                )
            else:
                payload = {
                    "schema_version": 1,
                    "run_id": run.get("run_id", run_id),
                    "title": title,
                    "question_count": len(questions),
                    "concepts": concepts,
                    "questions": questions,
                }
                archive.writestr(
                    "quiz-export/quiz.json",
                    json.dumps(payload, indent=2, ensure_ascii=False),
                )

        return zip_buffer.getvalue()

    def _questions_to_csv(self, questions: list[dict[str, Any]]) -> str:
        max_options = max((len(q.get("options") or []) for q in questions), default=0)
        option_columns = [f"option_{i + 1}" for i in range(max_options)]
        header = [
            "index",
            "difficulty",
            "target_concept",
            "question_text",
            *option_columns,
            "correct_answer",
            "explanation",
            "image",
        ]

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(header)
        for question in questions:
            options = list(question.get("options") or [])
            options += [""] * (max_options - len(options))
            writer.writerow(
                [
                    question.get("index", ""),
                    question.get("difficulty", ""),
                    question.get("target_concept", ""),
                    question.get("question_text", ""),
                    *options,
                    question.get("correct_answer", ""),
                    question.get("explanation", ""),
                    question.get("image", ""),
                ]
            )
        return buffer.getvalue()

    def _build_run_pdf(
        self,
        run: dict[str, Any],
        summary: dict[str, Any],
        results: list[dict[str, Any]],
        run_root: Path,
        title: str,
    ) -> bytes:
        import fitz  # PyMuPDF

        page_width, page_height = 595.0, 842.0  # A4 portrait
        margin_x, margin_top, margin_bottom = 50.0, 56.0, 56.0
        content_width = page_width - 2 * margin_x

        doc = fitz.open()
        state: dict[str, Any] = {"page": None, "y": margin_top}

        def new_page() -> None:
            state["page"] = doc.new_page(width=page_width, height=page_height)
            state["y"] = margin_top

        def ensure_space(height: float) -> None:
            if state["page"] is None or state["y"] + height > page_height - margin_bottom:
                new_page()

        def normalize_font(font: str) -> str:
            # Normalize legacy aliases to Base14 names supported by current PyMuPDF.
            mapping = {
                "helv": "helv",
                "helv-b": "hebo",
                "helv-oblique": "heit",
                "hebo": "hebo",
                "heit": "heit",
                "hebi": "hebi",
            }
            return mapping.get(font, "helv")

        def wrap(text: str, font: str, size: float, width: float) -> list[str]:
            draw_font = normalize_font(font)
            # Keep wrapping slightly conservative for wider variants.
            if draw_font == "hebo":
                eff_width = width * 0.92
            elif draw_font in ("heit", "hebi"):
                eff_width = width * 0.95
            else:
                eff_width = width

            cleaned = (text or "").replace("\r", "").strip()
            if not cleaned:
                return [""]
            lines: list[str] = []
            for raw_line in cleaned.split("\n"):
                if not raw_line.strip():
                    lines.append("")
                    continue
                words = raw_line.split()
                current = ""
                for word in words:
                    candidate = f"{current} {word}".strip() if current else word
                    if fitz.get_text_length(candidate, fontname=draw_font, fontsize=size) <= eff_width:
                        current = candidate
                        continue
                    if current:
                        lines.append(current)
                    if fitz.get_text_length(word, fontname=draw_font, fontsize=size) <= eff_width:
                        current = word
                    else:
                        chunk = ""
                        for ch in word:
                            tentative = chunk + ch
                            if fitz.get_text_length(tentative, fontname=draw_font, fontsize=size) <= eff_width:
                                chunk = tentative
                            else:
                                if chunk:
                                    lines.append(chunk)
                                chunk = ch
                        current = chunk
                if current:
                    lines.append(current)
            return lines or [""]

        def draw_lines(
            lines: list[str],
            font: str,
            size: float,
            *,
            indent: float = 0.0,
            color: tuple[float, float, float] = (0.1, 0.12, 0.16),
            line_height: float | None = None,
        ) -> None:
            step = line_height if line_height is not None else size * 1.35
            for line in lines:
                ensure_space(step)
                state["page"].insert_text(
                    (margin_x + indent, state["y"] + size),
                    line,
                    fontname=normalize_font(font),
                    fontsize=size,
                    color=color,
                )
                state["y"] += step

        def draw_text(
            text: str,
            font: str,
            size: float,
            *,
            indent: float = 0.0,
            color: tuple[float, float, float] = (0.1, 0.12, 0.16),
            line_height: float | None = None,
        ) -> None:
            draw_lines(
                wrap(text, font, size, content_width - indent),
                font,
                size,
                indent=indent,
                color=color,
                line_height=line_height,
            )

        def vertical_gap(amount: float) -> None:
            if state["page"] is None:
                return
            state["y"] = min(state["y"] + amount, page_height - margin_bottom)

        def find_correct_index(options: list[str], correct: str) -> int:
            cleaned = (correct or "").strip()
            if not cleaned or not options:
                return -1
            upper = cleaned.upper()
            if len(cleaned) == 1 and "A" <= upper <= "Z":
                idx = ord(upper) - ord("A")
                if 0 <= idx < len(options):
                    return idx
            for i, opt in enumerate(options):
                if str(opt).strip() == cleaned:
                    return i
            for i, opt in enumerate(options):
                opt_clean = str(opt).strip()
                if opt_clean.lower() == cleaned.lower():
                    return i
            if len(cleaned) >= 2 and "A" <= upper[0] <= "Z" and cleaned[1] in {".", ")", ":", "-"}:
                idx = ord(upper[0]) - ord("A")
                if 0 <= idx < len(options):
                    return idx
            for i, opt in enumerate(options):
                opt_clean = str(opt).strip().lower()
                if opt_clean.startswith(cleaned.lower()):
                    return i
            return -1

        new_page()
        draw_text(title, "helv-b", 20, color=(0.08, 0.1, 0.16))
        meta_line = f"{len(results)} questions"
        concepts = summary.get("concepts") if isinstance(summary.get("concepts"), list) else []
        if concepts:
            meta_line += " | " + ", ".join(str(c) for c in concepts[:6])
        draw_text(meta_line, "helv-oblique", 10, color=(0.4, 0.42, 0.5))
        vertical_gap(10)

        answer_entries: list[dict[str, Any]] = []

        for ordinal, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue
            question_text = str(item.get("question_text") or "").strip()
            options = [str(o) for o in (item.get("options") or []) if str(o)]
            correct = str(item.get("correct_answer") or "").strip()
            explanation = str(item.get("explanation") or "").strip()
            difficulty = str(item.get("difficulty") or "").strip()
            concept = str(item.get("target_concept") or "").strip()

            correct_idx = find_correct_index(options, correct)
            correct_letter = chr(ord("A") + correct_idx) if 0 <= correct_idx < len(options) else ""
            correct_text = options[correct_idx] if 0 <= correct_idx < len(options) else ""

            header_bits = [bit for bit in (difficulty, concept) if bit]
            header_suffix = f"  -  {' | '.join(header_bits)}" if header_bits else ""
            draw_text(
                f"Q{ordinal}{header_suffix}",
                "helv-b",
                11,
                color=(0.35, 0.38, 0.46),
            )
            vertical_gap(2)
            draw_text(question_text or "(no question text)", "helv", 12)
            vertical_gap(4)

            image_path = self._resolve_export_image_path(item.get("image_url", ""), run_root)
            if image_path:
                try:
                    img_doc = fitz.open(str(image_path))
                    if img_doc.page_count > 0:
                        img_page = img_doc.load_page(0)
                        img_rect = img_page.rect
                        max_w = min(content_width, 320.0)
                        max_h = 220.0
                        scale = min(max_w / img_rect.width, max_h / img_rect.height)
                        draw_w = img_rect.width * scale
                        draw_h = img_rect.height * scale
                        ensure_space(draw_h + 6)
                        target = fitz.Rect(
                            margin_x,
                            state["y"],
                            margin_x + draw_w,
                            state["y"] + draw_h,
                        )
                        state["page"].insert_image(target, filename=str(image_path), keep_proportion=True)
                        state["y"] += draw_h + 6
                    img_doc.close()
                except Exception:
                    pass

            for option_index, option in enumerate(options):
                letter = chr(ord("A") + option_index)
                lines = wrap(f"{letter}. {option}", "helv", 11, content_width - 14)
                draw_lines(lines, "helv", 11, indent=14, color=(0.18, 0.2, 0.26))

            vertical_gap(14)

            answer_entries.append(
                {
                    "ordinal": ordinal,
                    "letter": correct_letter,
                    "correct_text": correct_text or correct,
                    "raw_correct": correct,
                    "explanation": explanation,
                }
            )

        if answer_entries:
            new_page()
            draw_text("Answer key", "helv-b", 20, color=(0.08, 0.1, 0.16))
            draw_text(
                "Correct answers and explanations for every question.",
                "helv-oblique",
                10,
                color=(0.4, 0.42, 0.5),
            )
            vertical_gap(12)

            for entry in answer_entries:
                ordinal = entry["ordinal"]
                letter = entry["letter"]
                correct_text = entry["correct_text"]
                if letter and correct_text:
                    answer_line = f"Q{ordinal}.  {letter}.  {correct_text}"
                elif letter:
                    answer_line = f"Q{ordinal}.  {letter}"
                elif correct_text:
                    answer_line = f"Q{ordinal}.  {correct_text}"
                else:
                    answer_line = f"Q{ordinal}.  (no answer recorded)"
                draw_text(answer_line, "helv-b", 11, color=(0.13, 0.38, 0.24))
                if entry["explanation"]:
                    draw_text(
                        entry["explanation"],
                        "helv-oblique",
                        10,
                        indent=14,
                        color=(0.32, 0.34, 0.42),
                    )
                vertical_gap(10)

        buffer = io.BytesIO()
        doc.save(buffer, garbage=4, deflate=True)
        doc.close()
        pdf_bytes = buffer.getvalue()
        # region agent log
        try:
            import json
            import time as _time

            _path = self.settings.project_root / "debug-947ee0.log"
            with open(_path, "a", encoding="utf-8") as _lf:
                _lf.write(
                    json.dumps(
                        {
                            "sessionId": "947ee0",
                            "hypothesisId": "H1",
                            "location": "services.py:_build_run_pdf",
                            "message": "pdf_export_ok",
                            "data": {"bytes": len(pdf_bytes)},
                            "timestamp": int(_time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except OSError:
            pass
        # endregion
        return pdf_bytes

    def get_run_for_request(self, run_id: str, *, auth: AuthUser | None) -> dict[str, Any]:
        run = self.repo.get_run_by_run_id(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        notebook = self.repo.get_notebook(str(run.get("notebook_id", "")))
        if not notebook:
            raise HTTPException(status_code=404, detail="Run not found")
        self._require_notebook_owner(notebook, auth)
        hydrated = self._hydrate_run(run)
        if not hydrated:
            raise HTTPException(status_code=404, detail="Run not found")
        return hydrated

    def _api_artifact_href(self, absolute_file: Path) -> str:
        project_root = self.settings.project_root.resolve()
        resolved = absolute_file.resolve()
        relative = resolved.relative_to(project_root).as_posix()
        return f"/api/artifacts/{relative}"

    def _enrich_summary_results_with_disk_images(
        self, summary: dict[str, Any], manifest: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        """Fill empty per-question image_url from disk when the run output folder is present.

        Resolution order for each question index:
        1. ``image_url`` in ``generation/quiz_package.json`` for that index (relative to run root
           or http(s)/data URL), if the referenced file exists locally.
        2. ``generation/images/q{index}.<ext>`` or ``generation/images/{index}.<ext>`` under the run
           folder (``.png``, ``.jpg``, …).
        """
        results = summary.get("results")
        if not isinstance(results, list) or not results:
            return summary
        run_root = self._run_root_from_manifest(manifest, run_id)
        quiz_path = run_root / "generation" / "quiz_package.json"

        def normalize_question_index(value: object) -> int | None:
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
            return None

        disk_by_index: dict[int, str] = {}
        if quiz_path.is_file():
            try:
                payload = json.loads(quiz_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                payload = {}
            for item in payload.get("results", []) or []:
                if not isinstance(item, dict):
                    continue
                idx = normalize_question_index(item.get("index"))
                ref = item.get("image_url")
                if idx is not None and isinstance(ref, str) and ref.strip():
                    disk_by_index[idx] = ref.strip()

        image_dir = run_root / "generation" / "images"
        exts = (".png", ".jpg", ".jpeg", ".webp", ".gif")
        project_root = self.settings.project_root.resolve()

        def href_for_ref_on_disk(question_index: int) -> str:
            ref = disk_by_index.get(question_index)
            if ref:
                lowered = ref.lower()
                if lowered.startswith(("http://", "https://", "data:")):
                    return ref
                normalized = ref.replace("\\", "/").lstrip("/")
                try:
                    target = (run_root / normalized).resolve()
                except OSError:
                    target = run_root / normalized
                if (
                    project_root in target.parents or target == project_root
                ) and target.is_file():
                    return self._api_artifact_href(target)
            if image_dir.is_dir():
                for ext in exts:
                    for name in (f"q{question_index}{ext}", f"{question_index}{ext}"):
                        candidate = image_dir / name
                        try:
                            resolved = candidate.resolve()
                        except OSError:
                            continue
                        if resolved.is_file():
                            return self._api_artifact_href(resolved)
            return ""

        new_results: list[Any] = []
        changed = False
        hero = str(summary.get("hero_image") or "").strip()

        for row in results:
            if not isinstance(row, dict):
                new_results.append(row)
                continue
            row_out = dict(row)
            idx = normalize_question_index(row_out.get("index"))
            existing = str(row_out.get("image_url") or "").strip()
            if not existing and idx is not None:
                href = href_for_ref_on_disk(idx)
                if href:
                    row_out["image_url"] = href
                    changed = True
                    if not hero:
                        hero = href
            new_results.append(row_out)

        if not changed:
            return summary
        out = {**summary, "results": new_results}
        if hero and not str(summary.get("hero_image") or "").strip():
            out["hero_image"] = hero
        return out

    def _hydrate_run(self, run: dict[str, Any] | None) -> dict[str, Any] | None:
        if not run:
            return None
        summary = run.get("summary") or {}
        if isinstance(summary, str):
            try:
                summary = json.loads(summary)
            except json.JSONDecodeError:
                summary = {"raw": summary}
        manifest = run.get("manifest") or {}
        if isinstance(manifest, str):
            try:
                manifest = json.loads(manifest)
            except json.JSONDecodeError:
                manifest = {}
        artifact_paths = run.get("artifact_paths") or summary.get("artifact_paths", {})
        if isinstance(artifact_paths, str):
            try:
                artifact_paths = json.loads(artifact_paths)
            except json.JSONDecodeError:
                artifact_paths = {}
        run_id = str(run.get("run_id") or "")
        if isinstance(summary, dict) and isinstance(summary.get("results"), list):
            summary = self._enrich_summary_results_with_disk_images(summary, manifest, run_id)
        return {**run, "summary": summary, "manifest": manifest, "artifact_paths": artifact_paths}

    def _build_assistant_reply(
        self,
        prompt: str,
        *,
        sources: list[dict[str, Any]],
        runs: list[dict[str, Any]],
    ) -> str:
        normalized = prompt.lower()
        if not sources:
            return "There are no sources yet. Upload a PDF, markdown file, or text document first."
        if "quiz" in normalized and runs:
            latest = runs[0]
            summary = latest["summary"]
            return (
                f"The latest run has {summary.get('question_count', 0)} questions and covers "
                f"{', '.join(summary.get('concepts', [])[:4]) or 'the uploaded material'}."
            )
        if "source" in normalized:
            return f"This notebook currently has {len(sources)} source file(s) ready for generation."
        return (
            "Use Studio to generate a quiz, then review the question cards on the right. "
            "I keep the notebook state in Supabase-backed storage."
        )

    def _run_root_from_manifest(self, manifest: dict[str, Any], run_id: str) -> Path:
        raw_output_root = manifest.get("output_root") if isinstance(manifest, dict) else ""
        if isinstance(raw_output_root, str) and raw_output_root:
            candidate = self._safe_project_path(raw_output_root)
            if candidate and candidate.exists():
                return candidate
        return self.settings.output_root / run_id

    def _resolve_export_image_path(self, image_url: Any, run_root: Path) -> Path | None:
        raw = str(image_url or "").strip()
        if not raw:
            return None

        candidates: list[Path] = []
        parsed = urlparse(raw)
        if raw.startswith("/api/artifacts/"):
            artifact_path = unquote(raw.removeprefix("/api/artifacts/"))
            artifact_candidate = self._safe_project_path(artifact_path)
            if artifact_candidate:
                candidates.append(artifact_candidate)
        elif parsed.scheme in {"http", "https"}:
            filename = Path(unquote(parsed.path)).name
            if filename:
                candidates.append(run_root / "generation" / "images" / filename)
        else:
            normalized = unquote(raw).replace("\\", "/").lstrip("/")
            candidates.append(run_root / normalized)
            if Path(normalized).name:
                candidates.append(run_root / "generation" / "images" / Path(normalized).name)

        project_root = self.settings.project_root.resolve()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if project_root not in resolved.parents and resolved != project_root:
                continue
            if resolved.is_file():
                return resolved
        return None

    def _safe_project_path(self, value: str) -> Path | None:
        try:
            candidate = (self.settings.project_root / value).resolve()
            project_root = self.settings.project_root.resolve()
        except OSError:
            return None
        if project_root not in candidate.parents and candidate != project_root:
            return None
        return candidate

    def _export_image_name(self, ordinal: int, filename: str, used_names: set[str]) -> str:
        source_name = self._slugify(Path(filename).stem or f"image-{ordinal}")
        suffix = Path(filename).suffix.lower() or ".png"
        base_name = f"q{ordinal}_{source_name}{suffix}"
        name = f"images/{base_name}"
        counter = 2
        while name in used_names:
            name = f"images/q{ordinal}_{source_name}_{counter}{suffix}"
            counter += 1
        used_names.add(name)
        return name

    def _slugify(self, value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()
        return cleaned or "source"

    def _now(self) -> str:
        from .repository import utcnow_iso

        return utcnow_iso()

    def _now_token(self) -> str:
        return self._now().replace(":", "").replace("-", "").replace(".", "")
