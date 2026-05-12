from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


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
        project_root: Path,
        output_root: str | Path | None = None,
        run_id: str | None = None,
    ) -> "RunContext":
        root = Path(output_root) if output_root is not None else project_root / "outputs"
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


def _generate_run_id(document_path: Path) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{document_path.stem}_{uuid4().hex[:6]}"
