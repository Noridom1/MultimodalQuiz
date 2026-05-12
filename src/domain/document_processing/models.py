from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DocumentProcessingDomainResult:
    parsed_document: Any
    extracted: dict[str, Any]
    parsed_path: Path
    extracted_path: Path
