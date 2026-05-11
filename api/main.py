from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .repository import NotebookRepository
from .services import NotebookService


app = FastAPI(title="Multimodal Quiz UI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.allowed_origin, "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

repo = NotebookRepository(settings)
service = NotebookService(settings, repo)


class NotebookCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    description: str = ""


class MessageCreateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class GenerateQuizRequest(BaseModel):
    source_id: str
    num_questions: int = Field(default=5, ge=1, le=20)
    mock_image: bool = False
    mock_question: bool = False


class NotebookPatchRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)


class RunPatchRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


@app.get("/api/health")
def healthcheck() -> dict[str, object]:
    return {"ok": True, "supabase_enabled": settings.supabase_enabled}


@app.get("/api/notebooks")
def list_notebooks() -> list[dict[str, object]]:
    return service.list_notebook_cards()


@app.post("/api/notebooks")
def create_notebook(payload: NotebookCreateRequest) -> dict[str, object]:
    return repo.create_notebook(payload.title, payload.description)


@app.get("/api/notebooks/{notebook_id}")
def get_notebook(notebook_id: str) -> dict[str, object]:
    return service.get_workspace(notebook_id)


@app.patch("/api/notebooks/{notebook_id}")
def patch_notebook(notebook_id: str, payload: NotebookPatchRequest) -> dict[str, object]:
    return service.patch_notebook(notebook_id, title=payload.title)


@app.post("/api/notebooks/{notebook_id}/sources")
async def upload_source(
    notebook_id: str,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
) -> dict[str, object]:
    return await service.add_source(notebook_id, upload=file, title=title)


@app.post("/api/notebooks/{notebook_id}/messages")
def create_message(notebook_id: str, payload: MessageCreateRequest) -> dict[str, object]:
    return service.add_message(notebook_id, payload.content)


@app.post("/api/notebooks/{notebook_id}/generate")
def generate_quiz(notebook_id: str, payload: GenerateQuizRequest) -> dict[str, object]:
    return service.generate_quiz(
        notebook_id,
        source_id=payload.source_id,
        num_questions=payload.num_questions,
        mock_image=payload.mock_image,
        mock_question=payload.mock_question,
    )


@app.patch("/api/notebooks/{notebook_id}/runs/{run_id}")
def patch_run(notebook_id: str, run_id: str, payload: RunPatchRequest) -> dict[str, object]:
    return service.rename_run(notebook_id, run_id, payload.title)


@app.delete("/api/notebooks/{notebook_id}/runs/{run_id}")
def delete_run(notebook_id: str, run_id: str) -> dict[str, bool]:
    service.delete_notebook_run(notebook_id, run_id)
    return {"ok": True}


@app.get("/api/notebooks/{notebook_id}/runs/{run_id}/export")
def export_run(notebook_id: str, run_id: str) -> Response:
    payload, filename = service.export_run_zip(notebook_id, run_id)
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, object]:
    run = service._hydrate_run(repo.get_run_by_run_id(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/api/artifacts/{artifact_path:path}")
def get_artifact(artifact_path: str) -> FileResponse:
    target = (settings.project_root / artifact_path).resolve()
    project_root = settings.project_root.resolve()
    if project_root not in target.parents and target != project_root:
        raise HTTPException(status_code=403, detail="Forbidden path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(target)


frontend_dist = settings.project_root / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
