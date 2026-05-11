from __future__ import annotations

import io
import json
import mimetypes
import re
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from fastapi import HTTPException, UploadFile

from .config import Settings
from .repository import NotebookRepository
from src.pipeline import QuizGenerationPipeline


class NotebookService:
    def __init__(self, settings: Settings, repo: NotebookRepository) -> None:
        self.settings = settings
        self.repo = repo
        self.pipeline = QuizGenerationPipeline(output_root=settings.output_root)

    def list_notebook_cards(self) -> list[dict[str, Any]]:
        notebooks = self.repo.list_notebooks()
        cards: list[dict[str, Any]] = []
        for notebook in notebooks:
            workspace = self.get_workspace(notebook["id"])
            cards.append(
                {
                    **workspace["notebook"],
                    "source_count": len(workspace["sources"]),
                    "run_count": len(workspace["runs"]),
                    "question_count": workspace["latest_run"]["summary"].get("question_count", 0)
                    if workspace["latest_run"]
                    else 0,
                    "hero_image": workspace["notebook"].get("cover_image")
                    or (workspace["latest_run"]["summary"].get("hero_image") if workspace["latest_run"] else ""),
                }
            )
        return cards

    def get_workspace(self, notebook_id: str) -> dict[str, Any]:
        notebook = self.repo.get_notebook(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

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
    ) -> dict[str, Any]:
        notebook = self.repo.get_notebook(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

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

    def add_message(self, notebook_id: str, content: str) -> dict[str, Any]:
        notebook = self.repo.get_notebook(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

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
    ) -> dict[str, Any]:
        notebook = self.repo.get_notebook(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

        source = next((item for item in self.repo.list_sources(notebook_id) if item["id"] == source_id), None)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        document_path = Path(source["local_path"])
        if not document_path.exists():
            raise HTTPException(status_code=400, detail="Source file is no longer available locally")

        pending_run = self.repo.create_run(
            {
                "notebook_id": notebook_id,
                "source_id": source_id,
                "run_id": f"pending-{self._slugify(notebook['title'])}-{self._now_token()}",
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
            result = self.pipeline.run(
                document_path,
                num_questions=num_questions,
                mock_image=mock_image,
                mock_question=mock_question,
            )
            run_id = result["run_id"]
            manifest_path = Path(result["manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary = self._build_run_summary(notebook_id, result)
            updated = self.repo.update_run(
                pending_run["run_id"],
                {
                    "run_id": run_id,
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
            return self._hydrate_run(updated or self.repo.get_run_by_run_id(run_id))
        except Exception as exc:
            self.repo.update_run(pending_run["run_id"], {"status": "failed", "summary": {"error": str(exc)}})
            self.repo.create_message(
                notebook_id,
                "assistant",
                f"Quiz generation failed: {exc}",
                kind="system",
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    def patch_notebook(self, notebook_id: str, *, title: str | None) -> dict[str, Any]:
        notebook = self.repo.get_notebook(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
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

    def rename_run(self, notebook_id: str, run_id: str, title: str) -> dict[str, Any]:
        run = self.repo.get_run_by_run_id(run_id)
        if not run or run.get("notebook_id") != notebook_id:
            raise HTTPException(status_code=404, detail="Run not found")
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

    def delete_notebook_run(self, notebook_id: str, run_id: str) -> None:
        run = self.repo.get_run_by_run_id(run_id)
        if not run or run.get("notebook_id") != notebook_id:
            raise HTTPException(status_code=404, detail="Run not found")
        self.repo.delete_run(run_id)

    def export_run_zip(self, notebook_id: str, run_id: str) -> tuple[bytes, str]:
        run = self._hydrate_run(self.repo.get_run_by_run_id(run_id))
        if not run or run.get("notebook_id") != notebook_id:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Only completed quizzes can be exported")

        summary = run.get("summary") or {}
        results = summary.get("results") or []
        if not isinstance(results, list) or not results:
            raise HTTPException(status_code=400, detail="Run has no quiz questions to export")

        run_root = self._run_root_from_manifest(run.get("manifest") or {}, run_id)
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

            payload = {
                "schema_version": 1,
                "run_id": run.get("run_id", run_id),
                "title": summary.get("title") or run.get("title") or "Untitled quiz",
                "question_count": len(questions),
                "concepts": summary.get("concepts", []) if isinstance(summary.get("concepts"), list) else [],
                "questions": questions,
            }
            archive.writestr(
                "quiz-export/quiz.json",
                json.dumps(payload, indent=2, ensure_ascii=False),
            )

        filename = f"{self._slugify(str(payload['title'])) or self._slugify(run_id)}.zip"
        return zip_buffer.getvalue(), filename

    def _build_run_summary(self, notebook_id: str, result: dict[str, Any]) -> dict[str, Any]:
        run_root = Path(result["run_root"])
        quiz_package_path = run_root / "generation" / "quiz_package.json"
        quiz_package = json.loads(quiz_package_path.read_text(encoding="utf-8"))
        published = self._publish_run_artifacts(notebook_id, result["run_id"], run_root, quiz_package)
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
            "title": f"Run {result['run_id']}",
            "question_count": len(quiz_package.get("results", [])),
            "hero_image": hero_image,
            "concepts": concepts,
            "results": [
                {
                    "index": item["index"],
                    "question_text": item["question"]["question_text"],
                    "difficulty": item["question"]["difficulty"],
                    "target_concept": item["question"]["target_concept"],
                    "explanation": item["question"]["explanation"],
                    "correct_answer": item["question"]["correct_answer"],
                    "options": item["question"].get("options", []),
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
