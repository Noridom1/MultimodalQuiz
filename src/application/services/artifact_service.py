from __future__ import annotations

from pathlib import Path
from typing import Any

from src.application.services.run_context import RunContext
from src.utils.io import relative_path, write_json


class RunArtifactService:
    def __init__(self, *, project_root: Path) -> None:
        self._project_root = project_root

    def project_relative(self, path: Path) -> str:
        return relative_path(path, self._project_root)

    def rename_artifact(self, source: Path, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() == target.resolve():
            return target
        if target.exists():
            target.unlink()
        source.replace(target)
        return target

    def write_manifest(self, context: RunContext, manifest: dict[str, Any]) -> None:
        write_json(context.manifest_path, manifest)
